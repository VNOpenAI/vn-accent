python3 train.py 'data/tokenizer.h5' 'data/train' 'data/val' \
--num_epochs 20 \
--cuda \
--learning_rate 0.0003 \
--model_name 'base_evolved' \
--weight_dir 'base_weight'
