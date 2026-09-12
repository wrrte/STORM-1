#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

python eval_frostbite_value.py   --baseline-run Frostbite-life_done-wm_2L512D8H-100k-seed710_X   --flash-run Frostbite-life_done-wm_2L512D8H-100k-seed710_O   --step 100000 --output results/seed_710
python eval_frostbite_value.py   --baseline-run Frostbite-life_done-wm_2L512D8H-100k-seed10_X   --flash-run Frostbite-life_done-wm_2L512D8H-100k-seed10_O   --step 100000 --output results/seed_10
python eval_frostbite_value.py   --baseline-run Frostbite-life_done-wm_2L512D8H-100k-seed3710   --flash-run Frostbite-life_done-wm_2L512D8H-100k-seed3710-new_O   --step 100000 --output results/seed_3710
python eval_frostbite_value.py   --baseline-run Frostbite-life_done-wm_2L512D8H-100k-seed1_X   --flash-run Frostbite-life_done-wm_2L512D8H-100k-seed1_O   --step 100000 --output results/seed_1



stat -c '%n | 마지막 수 정: %y'   /media/storage_data/ai2lab/choemj/STORM/ckpt/Frostbite-life_done-wm_2L512D8H-100k-seed3710/*_100000.pth
stat -c '%n | 마지막 수 정: %y'   /media/storage_data/ai2lab/choemj/STORM/ckpt/Frostbite-life_done-wm_2L512D8H-100k-seed3710_X/*_100000.pth
stat -c '%n | 마지막 수 정: %y'   /media/storage_data/ai2lab/choemj/STORM/ckpt/Frostbite-life_done-wm_2L512D8H-100k-seed710_X/*_100000.pth
stat -c '%n | 마지막 수 정: %y'   /media/storage_data/ai2lab/choemj/STORM/ckpt/Frostbite-life_done-wm_2L512D8H-100k-seed1_X/*_100000.pth
stat -c '%n | 마지막 수 정: %y'   /media/storage_data/ai2lab/choemj/STORM/ckpt/Frostbite-life_done-wm_2L512D8H-100k-seed3710-new_O/*_100000.pth
stat -c '%n | 마지막 수 정: %y'   /media/storage_data/ai2lab/choemj/STORM/ckpt/Frostbite-life_done-wm_2L512D8H-100k-seed3710_O/*_100000.pth
stat -c '%n | 마지막 수 정: %y'   /media/storage_data/ai2lab/choemj/STORM/ckpt/Frostbite-life_done-wm_2L512D8H-100k-seed710_O/*_100000.pth
stat -c '%n | 마지막 수 정: %y'   /media/storage_data/ai2lab/choemj/STORM/ckpt/Frostbite-life_done-wm_2L512D8H-100k-seed1_O/*_100000.pth
