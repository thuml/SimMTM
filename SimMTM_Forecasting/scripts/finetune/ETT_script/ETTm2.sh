xport CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm2 \
        --model SimMTM \
        --data ETTm2 \
        --features M \
        --seq_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 16 \
        --dropout 0 \
        --batch_size 64
done

