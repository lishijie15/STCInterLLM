model_path=./checkpoints/vicuna-7b-v1.5-16k
instruct_ds=./ST_data/train_10pv/train_10pv.json  #multi_NYC.json
st_data_path=./ST_data/train_10pv/train_10pv.pkl      #multi_NYC_pkl.pkl
pretra_ste=Causal_Encoder
output_model=./checkpoints/Causal_Encoder_7b_pv10_noNorm

wandb offline
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master_port=20001 \
    STCILLM/train/train_power.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --st_content ./TAXI.json \
    --st_data_path ${st_data_path} \
    --st_tower ${pretra_ste} \
    --tune_st_mlp_adapter True \
    --st_select_layer -2 \
    --use_st_start_end \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4800 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb

