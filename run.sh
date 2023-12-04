#!/bin/bash
set -x

NEMO="./NeMo"
MAX_SEQ_LEN=8192
MAX_ITERS=5000
MBS=1
TP=1
PP=8
NUM_DEVICES=8

MODEL=./model/lama-2-70b.nemo
EXP_DIR=./results

TRAIN_DS=[./dataset/train.jsonl]
VALID_DS=[./dataset/test.jsonl]

export PYTHONPATH="${NEMO}/.:${PYTHONPATH}"
torchrun --nproc_per_node=${NUM_DEVICES} ${NEMO}/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
        trainer.devices=${NUM_DEVICES} \
        trainer.num_nodes=1 \
        trainer.precision=16 \
        trainer.val_check_interval=50 \
        trainer.max_steps=${MAX_ITERS} \
        ++trainer.limit_val_batches=50 \
        model.megatron_amp_O2=False \
        ++model.mcore_gpt=True \
        +model.recompute_activations=True \
        exp_manager.exp_dir="${EXP_DIR}" \
        model.tensor_model_parallel_size=${TP} \
        model.pipeline_model_parallel_size=${PP} \
        model.sequence_parallel=True \
        model.micro_batch_size=${MBS} \
        model.global_batch_size=1 \
        model.restore_from_path=${MODEL} \
        model.data.train_ds.num_workers=0 \
        model.data.validation_ds.num_workers=0 \
        model.data.validation_ds.metric.name=exact_string_match \
        model.data.validation_ds.tokens_to_generate=4 \
        model.data.train_ds.file_names=${TRAIN_DS} \
        model.data.train_ds.concat_sampling_probabilities=[1.0] \
        model.data.validation_ds.file_names=${VALID_DS} \
        model.peft.peft_scheme=lora \
        model.data.train_ds.max_seq_length=${MAX_SEQ_LEN} \
        model.data.validation_ds.max_seq_length=${MAX_SEQ_LEN}