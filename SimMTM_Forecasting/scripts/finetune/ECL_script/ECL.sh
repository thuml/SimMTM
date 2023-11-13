export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720; do
    python -u run.py \
        --task_name finetune \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id ECL \
        --model SimMTM \
        --data ECL \
        --features M \
        --seq_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --d_model 32 \
        --d_ff 64 \
        --n_heads 16 \
        --batch_size 32
done


