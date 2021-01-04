python3 train.py 'data/tokenizer.h5' 'data/train' 'data/val' \
--num_epochs 20 \
--cuda \
--learning_rate 0.001 \
--model_name 'bi_lstm' \
--experiment_name 'bi_lstm'
