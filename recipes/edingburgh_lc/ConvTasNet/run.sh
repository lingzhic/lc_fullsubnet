#!/bin/bash

# Exit on error
set -e
set -o pipefail

# General
python_path=python 	# Path to the python you'll use for the experiment. Defaults to the current python
stage=1  # Controls from which stage to start
tag=""   # Controls the directory name associated to the experiment
id=$CUDA_VISIBLE_DEVICES # Ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
# Evaluation cofig
eval_use_gpu=1
out_dir=evaluation_output # Controls the directory name associated to the evaluation results inside the experiment directory
# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

. utils/parse_options.sh
# ============================================ Args ============================================

train_path=edingburgh_meta/edingburgh_train_meta.csv
valid_path=edingburgh_meta/edingburgh_valid_meta.csv
test_path=edingburgh_meta/edingburgh_test_meta.csv

expdir=exp/lc_convtasnet_${tag}
mkdir -p $expdir
echo "Results from the following experiment will be stored in $expdir"

# ============================== Training ==============================
if [[ $stage -le 1 ]]; then
	echo "Stage 1: Training"
	CUDA_VISIBLE_DEVICES=$id $python_path lc_train.py --exp_dir $expdir | tee $expdir/train_${tag}.log
fi
# ============================== Training ==============================

# ============================== Evaluation ==============================
if [[ $stage -le 2 ]]; then
	echo "Stage 2 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path lc_eval.py --exp_dir $expdir \
													 --test_path $test_path \
													 --out_dir $out_dir \
													 --use_gpu $eval_use_gpu | tee $expdir/eval_${tag}.log
fi
# ============================== Evaluation ==============================
