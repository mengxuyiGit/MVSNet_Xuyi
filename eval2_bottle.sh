#!/usr/bin/env bash
BOTTLE_TESTING="/mnt/lustre/share/yslan/bottles"
CKPT_FILE="./checkpoint/model_000014.ckpt"
python eval2_bottle.py  \
--dataset=bottle_eval \
--batch_size=1 \
--testpath=$BOTTLE_TESTING \
--testlist metadata/bottles/test.txt \
--metapath metadata/bottles \
--loadckpt $CKPT_FILE \
--numdepth 384 \
--interval_scale 0.01 \
--depth_min 1.5 \
--outdir 'outputs_final' $@

# pair_data = pair_0_train & pair_1_val.txt (both are the final manually selected pairs of 100 scans)