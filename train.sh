#!/usr/bin/env bash

python -u cli.py --do_train --output_dir output \
        --train_file datasets_v1/generated_sharc_dialogues_sample_train.json \
        --predict_file datasets_v1/generated_sharc_dialogues_dev.json \
        --BART facebook/bart-large \
        --train_batch_size 16 \
        --predict_batch_size 16 \
        --eval_period 100 \
        --append_another_bos
        