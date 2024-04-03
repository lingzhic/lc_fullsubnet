#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Path to the python you'll use for the experiment. Defaults to the current python
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=1  # Controls from which stage to start
tag=""   # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES
out_dir=edingburgh_lc # Controls the directory name associated to the evaluation results inside the experiment directory

# Network config
n_blocks=8		# (int, optional): Number of convolutional blocks in each repeat. Defaults to 8.
n_repeats=3		# (int, optional): Number of repeats. Defaults to 3.
mask_act=relu
causal=true
# Training config
epochs=1
batch_size=8
num_workers=4
half_lr=true
early_stop=true
# Optim config
optimizer=adam
lr=0.001
weight_decay=0.
# Data config
target_sr=16000
segment=1		# segment (int, optional) : The desired sources and mixtures length in s.
# Evaluation cofig
eval_use_gpu=1
compute_wer=0	# Need to --compute_wer 1 --eval_mode max to be sure the user knows all the metrics are for the all mode.
eval_mode=
task=enh_single

. utils/parse_options.sh
# ============================================ Args ============================================

if [ -z "$eval_mode" ]; then
  eval_mode=$mode
fi

train_path=edingburgh_meta/edingburgh_train_meta.csv
valid_path=edingburgh_meta/edingburgh_valid_meta.csv
test_path=edingburgh_meta/edingburgh_test_meta.csv

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi

expdir=exp/lc_convtasnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"


if [[ $stage -le 1 ]]; then
  echo "Stage 1: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path lc_train.py --exp_dir $expdir \
		--n_blocks $n_blocks \
		--n_repeats $n_repeats \
		--mask_act $mask_act \
		--epochs $epochs \
		--batch_size $batch_size \
		--num_workers $num_workers \
		--half_lr $half_lr \
		--early_stop $early_stop \
		--optimizer $optimizer \
		--lr $lr \
		--weight_decay $weight_decay \
		--train_path $train_path \
		--valid_path $valid_path \
		--target_sr $target_sr \
		--segment $segment | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log
fi


if [[ $stage -le 2 ]]; then
	echo "Stage 2 : Evaluation"

	if [[ $compute_wer -eq 1 ]]; then
		if [[ $eval_mode != "max" ]]; then
		echo "Cannot compute WER without max mode. Start again with --stage 1 --compute_wer 1 --eval_mode max"
		exit 1
		fi

		# Install espnet if not instaled
		if ! python -c "import espnet" &> /dev/null; then
			echo 'This recipe requires espnet. Installing requirements.'
			$python_path -m pip install espnet_model_zoo
			$python_path -m pip install jiwer
			$python_path -m pip install tabulate
		fi
	fi

	$python_path lc_eval.py \
		--exp_dir $expdir \
		--test_path $test_path \
		--out_dir $out_dir \
		--use_gpu $eval_use_gpu \
		--compute_wer $compute_wer \
		--task $task | tee logs/eval_${tag}.log

	cp logs/eval_${tag}.log $expdir/eval.log
fi
