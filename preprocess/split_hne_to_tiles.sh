#!/bin/bash

split_hne_patches(){
    python split_hne_to_patches.py  --hne /home/jupyter/Data/CRC_Lin/WD-76845-$1.ome.tif --out_path /home/jupyter/CycifPreprocess/Crops/WD-76845-$1 --out_file_prefix WD-76845-$1 --patch_size 256 --step 128 &
}


hne_nums=(001 006 013 019 024 028 033 038 043 048 053)
for i in "${!hne_nums[@]}"; do
     split_hne_patches ${hne_nums[$i]}
done
wait

hne_nums=(058 063 068 073 077 083 085 090 096 101 105)
for i in "${!hne_nums[@]}"; do
     split_hne_patches ${hne_nums[$i]}
done
wait

split_hne_patches2(){
    nohup python split_hne_to_patches.py  --hne /home/jupyter/Data/CRC_Lin/CRC$1-HE.ome.tif --out_path /home/jupyter/CycifPreprocess/Crops/CRC$1 --out_file_prefix CRC$1 --patch_size 256 --step 128 &
}

hne_nums=(02 03 04 05 06 07 08 09)
for i in "${!hne_nums[@]}"; do
     split_hne_patches2 ${hne_nums[$i]}
done
wait

hne_nums=(10 11 12 13 14 15 16 17)
for i in "${!hne_nums[@]}"; do
     split_hne_patches2 ${hne_nums[$i]}
done
wait
