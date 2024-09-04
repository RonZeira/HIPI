#!/bin/bash
######################################## Download CRC atlas data ###################################################
## Download all preprocessed CRC atlas data
## s3://lin-2021-crc-atlas/data/


download_cycif_features() {
    wget --no-check-certificate --content-disposition https://lin-2021-crc-atlas.s3.amazonaws.com/data/$1-features.zip -P .
    unzip $1-features.zip
    rm $1-features.zip
}

download_image_files() {
    wget --no-check-certificate --content-disposition https://lin-2021-crc-atlas.s3.amazonaws.com/data/$1.ome.tif -P .
}

for fn in WD-76845-002 WD-76845-007 WD-76845-014 WD-76845-020 WD-76845-025 WD-76845-029 WD-76845-034 WD-76845-039 WD-76845-044 WD-76845-049 WD-76845-050 WD-76845-051 WD-76845-052 WD-76845-054 WD-76845-059 WD-76845-064 WD-76845-069 WD-76845-074 WD-76845-078 WD-76845-084 WD-76845-086 WD-76845-091 WD-76845-097 WD-76845-102 WD-76845-106 CRC02 CRC03 CRC04 CRC05 CRC06 CRC07 CRC08 CRC09 CRC10 CRC11 CRC12 CRC13 CRC14 CRC15 CRC16 CRC17; do
    download_cycif_features ${fn} &
done

wait
echo | ls *.csv
echo "Done CSVs"

for fn in WD-76845-001 WD-76845-002 WD-76845-006 WD-76845-007 WD-76845-013 WD-76845-014 WD-76845-019 WD-76845-020 WD-76845-024 WD-76845-025 WD-76845-028 WD-76845-029 WD-76845-033 WD-76845-034 WD-76845-038 WD-76845-039 WD-76845-043 WD-76845-044 WD-76845-048 WD-76845-049 WD-76845-050 WD-76845-051 WD-76845-052 WD-76845-053 WD-76845-054 WD-76845-058 WD-76845-059 WD-76845-063 WD-76845-064 WD-76845-068 WD-76845-069 WD-76845-073 WD-76845-074 WD-76845-077 WD-76845-078 WD-76845-083 WD-76845-084 WD-76845-085 WD-76845-086 WD-76845-090 WD-76845-091 WD-76845-096 WD-76845-097 WD-76845-101 WD-76845-102 WD-76845-105 WD-76845-106 CRC02 CRC03 CRC04 CRC05 CRC06 CRC07 CRC08 CRC09 CRC10 CRC11 CRC12 CRC13 CRC14 CRC15 CRC16 CRC17; do
    download_image_files ${fn} &
done

for fn in CRC02 CRC03 CRC04 CRC05 CRC06 CRC07 CRC08 CRC09 CRC10 CRC11 CRC12 CRC13 CRC14 CRC15 CRC16 CRC17; do
    wget --no-check-certificate --content-disposition https://lin-2021-crc-atlas.s3.amazonaws.com/data/${fn}-HE.ome.tif -P . &
done

wait
echo | ls *.tif
echo "Done TIFs"
