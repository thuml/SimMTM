export CUDA_VISIBLE_DEVICES=0

python -u run.py \
    --task_name pretrain \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1 \
    --model SimMTM \
    --data ETTm1 \
    --features M \
    --seq_len 336 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 8 \
    --d_model 32 \
    --d_ff 64 \
    --positive_nums 3 \
    --mask_rate 0.5 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 50

