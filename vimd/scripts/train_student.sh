#!/bin/bash

CUDA_VISIBLE_DEVICES=1 torchrun --master_port=8089 --nproc_per_node=1 train_student.py  \
--model vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 16  \
--lr 5e-7 --input-size 224 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --num_workers 1  \
--data-set DOG --data-path /home/hello/Documents/datasets/DOG  \
--output_dir /home/hello/code/SR_MAMBA/script/DOG/vimtiny_SRGAN_DOG_HiddenStates24Mse  \
--epochs 250 --finetune ./Vim-tiny-midclstok/vim_t_midclstok_ft_78p3acc.pth --no_amp
