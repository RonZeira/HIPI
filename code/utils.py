"""
File: utils.py
Author: Ron Zeira
Description: This file contains the utility functions for the HIPI project.
"""

import numpy as np
import torch
import importlib
import omegaconf

def model_params(model, trainable = True):
    model_parameters = filter(lambda p: not(trainable^p.requires_grad), model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def instantiate_from_config_extra_args(config, *args, **kwargs):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(*args, **config.get("params", dict()), **kwargs)

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def get_model_from_function_or_cfg(model, **kwargs):
    if isinstance(model, torch.nn.Module):
        return model
    elif isinstance(model, dict) or isinstance(model, omegaconf.dictconfig.DictConfig):
        return instantiate_from_config(model)
    elif callable(model):
        return model(**kwargs)
    else:
        raise ValueError('Unknown type for model', type(model))
    
def get_pytorch_layer(s):
    if callable(s):
        return s
    elif s is None:
        return None
    elif not isinstance(s,str):
        raise ValueError('Unknown type', s)
    try:
        f = eval(s)
    except NameError:
        try:
            f = eval('torch.nn.'+s)
        except NameError:
            raise NameError('Unknown '+s)
    return f

def load_model_from_config(config, sd, device = None):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    if device:
        model = model.to(device)
    model.eval()
    return model

def load_model(config, ckpt, device = None):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])
    return model, global_step