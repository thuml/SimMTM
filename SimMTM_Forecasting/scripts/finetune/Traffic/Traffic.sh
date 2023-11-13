export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --root_path ./dataset/traffic/ \
        --data_path traffic.csv \
        --model_id Traffic \
        --model SimMTM \
        --data Traffic \
        --features M \
        --seq_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --d_model 128 \
        --d_ff 256 \
        --n_heads 16 \
        --batch_size 32 \
        --dropout 0.2
done


