"""
File: models.py
Author: Ron Zeira
Description: This file contains the model classes and functions for the HIPI project.
"""

import torch, torchvision, timm, os, sys
from functools import partial
import pytorch_lightning as pl
# sys.path.append('../../HIPT/HIPT/HIPT_4K/')
# import vision_transformer as vits_hipt
from .utils import *

############################################################################

### ResNet trunk feature extractor
class SslResNetTrunk(torchvision.models.resnet.ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

### ssl pathology model getter from https://github.com/lunit-io/benchmark-ssl-pathology
def ssl_get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url

### ssl resnet feature extractor from pretrained
def ssl_resnet50(pretrained, progress, key, **kwargs):
    model = SslResNetTrunk(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = ssl_get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model

### ssl vit feature extractor from pretrained
def ssl_vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = timm.models.vision_transformer.VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = ssl_get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model

### ssl HIPT feature extractor from pretrained
def get_HIPT_vit256(pretrained_weights, device=torch.device('cpu'), img_size=256):
    checkpoint_key = 'teacher'
    # model256 = vits_hipt.__dict__[arch](patch_size=16, num_classes=0)
    model256 = timm.models.vision_transformer.VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, img_size = img_size,
        qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_classes=0)
    for p in model256.parameters():
        p.requires_grad = False
    model256.eval()
    model256.to(device)

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model256.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        
    return model256

############################################################################

### feature extractor network wrapper class
class pretrained_net_wrapper(torch.nn.Module):
    def __init__(self, cls_func, model_im_size, resize_to_model = False, freeze_model = True, *args, **kwargs):
        super().__init__()
        self.net = cls_func(*args, **kwargs)
        if freeze_model:
            for p in self.net.parameters():
                p.requires_grad = False
        self.model_im_size = model_im_size
        self.resize_to_model = resize_to_model
    def forward(self,x):
        if self.resize_to_model and x.shape[-1] != self.model_im_size:
            x = torchvision.transforms.Resize(self.model_im_size, antialias=True)(x)
        return self.net(x)

### ssl resnet50 wrapper class
class ssl_resnet50_wrapper(pretrained_net_wrapper):
    def __init__(self, pretrained, progress, key, *args, **kwargs):
        super().__init__(pretrained = pretrained, progress = progress, key = key, 
                         cls_func = ssl_resnet50, model_im_size = None, resize_to_model = False, 
                         *args, **kwargs)
        self.out_features = 2048
    def forward(self,x):
        z = super().forward(x)
        return self.net.avgpool(z).squeeze(-1).squeeze(-1)

### ssl vit wrapper class
class ssl_vit_small_wrapper(pretrained_net_wrapper):
    def __init__(self, pretrained, progress, key, *args, **kwargs):
        super().__init__(pretrained = pretrained, progress = progress, key = key,
                         cls_func = ssl_vit_small, model_im_size = 224, resize_to_model = True, 
                         *args, **kwargs)
        self.out_features = 384

### ssl resnet50 (imagenet) wrapper class
class imagenet_resnet_wrapper(pretrained_net_wrapper):
    def __init__(self, resnet_cls = torchvision.models.resnet50, output_layer = 'avgpool', weights = torchvision.models.ResNet50_Weights.DEFAULT, *args, **kwargs):
        super().__init__(cls_func = resnet_cls, model_im_size = None, resize_to_model = False,
                         weights = torchvision.models.ResNet50_Weights.DEFAULT, 
                         *args, **kwargs)
        self.out_features = 2048
        layers = list(self.net._modules.keys())
        if output_layer in layers:
            self.net = torch.nn.Sequential(*[self.net._modules[layers[i]] for i in range(layers.index(output_layer)+1)])
    def forward(self,x):
        return super().forward(x).squeeze(-1).squeeze(-1)

### ssl hipt256 wrapper class
class hipt256_wrapper(pretrained_net_wrapper):
    def __init__(self, pretrained_weights, device=torch.device('cpu'), model_im_size=256, *args, **kwargs):
        super().__init__(cls_func = get_HIPT_vit256, model_im_size = model_im_size, resize_to_model = False, 
                         pretrained_weights = pretrained_weights, device= device, img_size = model_im_size, 
                         *args, **kwargs)
        self.out_features = 384

# resnet50_model = ssl_resnet50_wrapper(pretrained=True, progress=False, key="BT")
# vit_model = ssl_vit_small_wrapper(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
# imagenet_resnet50_model = imagenet_resnet_wrapper()
# hipt_model256 = hipt256_wrapper(pretrained_weights='../HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth', model_im_size = 256)

class image_channel_average(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.out_features = 3
    def forward(self,x):
        return x.mean(dim=[-2,-1])

### Loss class for the HIPI model with logging
class cycif_loss(torch.nn.Module):
    def __init__(self, loss = torch.nn.MSELoss, channel_names = None, weights = None):
        super().__init__()
        if isinstance(loss, list) or isinstance(loss, tuple):
            self.loss_fn = [get_pytorch_layer(l) for l in loss]
        else:
            self.loss_fn = get_pytorch_layer(loss)
        self.weights = weights
        self.channel_names = channel_names
    
    def forward(self, pred, y, split = 'train'):
        if isinstance(self.loss_fn, list) or isinstance(self.loss_fn, tuple):
            loss = torch.concat([self.loss_fn[i](reduction='none')(pred[:,i], y[:,i]).unsqueeze(1) for i in range(len(self.loss_fn))], dim=1)
        else:
            loss = self.loss_fn(reduction='none')(pred, y)
        per_channel_loss = loss.mean(dim = 0)
        if self.weights:
            per_channel_loss = per_channel_loss * torch.tensor(self.weights)
        
        log = {f"{split}/{i if not self.channel_names else self.channel_names[i]}_loss": per_channel_loss[i].clone().detach().mean() for i in range(per_channel_loss.shape[0])}
        log[f"{split}/total_loss"] = per_channel_loss.clone().detach().mean()
        return per_channel_loss.mean(), log

### HIPI model class composed of a feature extractor and a regressor. Incldes the train/eval loops.
class cycif_image_regressor(pl.LightningModule):
    def __init__(self, feature_extractor, out_channels, hidden_channels = [],
                norm_layer = torch.nn.BatchNorm1d, activation_layer = torch.nn.GELU, dropout = 0.0, 
                final_activation = None, freeze_feature_extractor = True, base_learning_rate = 1e-5,
                ckpt_path=None, ignore_keys=[], monitor=None,
                loss = cycif_loss, optimizer = None, scheduler = None,
                 *args, **kwargs):
        # super().__init__()
        pl.LightningModule.__init__(self)
        self.feature_extractor = get_model_from_function_or_cfg(feature_extractor)
        self.out_channels = out_channels
        self.mlp = torchvision.ops.misc.MLP(in_channels = self.feature_extractor.out_features, 
                                            hidden_channels = hidden_channels + [out_channels], 
                                            norm_layer = get_pytorch_layer(norm_layer), activation_layer = get_pytorch_layer(activation_layer), 
                                            dropout = dropout)
        # delete the last dropout
        del self.mlp[len(self.mlp)-1]
        if isinstance(final_activation, list) or isinstance(final_activation, tuple):
            self.final_activation = [get_pytorch_layer(act) for act in final_activation]
        elif final_activation is None:
            self.final_activation = final_activation
        else:
            self.final_activation = get_pytorch_layer(final_activation)
        if isinstance(self.final_activation, list) or isinstance(self.final_activation, tuple):
            assert(len(self.final_activation) == out_channels)
        self.loss = get_model_from_function_or_cfg(loss)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.freeze_feature_extractor = freeze_feature_extractor
        self.learning_rate = base_learning_rate
        for p in self.feature_extractor.parameters():
            p.requires_grad = not self.freeze_feature_extractor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if monitor is not None:
            self.monitor = monitor
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        
    def forward(self, x):
        y = self.feature_extractor(x)
        y = self.mlp(y)
        if self.final_activation:
            if isinstance(self.final_activation, list) or isinstance(self.final_activation, tuple):
                y = torch.concat([self.final_activation[i]()(y[:,i]).unsqueeze(1) for i in range(self.out_channels)], dim=1)
            else:
                y = self.final_activation()(y)
        return y
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.mlp.parameters())
        if not self.freeze_feature_extractor:
            params = params + list(self.feature_extractor.parameters())
        if self.optimizer is None: 
            opt_ae = torch.optim.Adam(params,
                                    lr=lr, betas=(0.5, 0.9))
            # opt_ae = torch.optim.sgd(params, lr=lr, momentum = 0.9)
        else:
            opt_ae = instantiate_from_config_extra_args(self.optimizer, params, lr = lr)
        if self.scheduler is None: 
            return opt_ae
        else:
            opt_scheduler = instantiate_from_config_extra_args(self.scheduler, opt_ae)
            return [opt_ae], [opt_scheduler]
    
    def general_step(self, batch, batch_idx = -1, optimizer_idx = 0, split = "train"):
        imgs, labels = batch
        predictions = self(imgs)
        loss, log_dict = self.loss(predictions, labels, split=split)          
        self.log(f'{split}/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return loss
    
    def training_step(self, batch, batch_idx = -1, optimizer_idx = 0):
        return self.general_step(batch, batch_idx = batch_idx, optimizer_idx = optimizer_idx, split = "train")
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx = -1, optimizer_idx = 0):
        return self.general_step(batch, batch_idx = batch_idx, optimizer_idx = optimizer_idx, split = "val")
    