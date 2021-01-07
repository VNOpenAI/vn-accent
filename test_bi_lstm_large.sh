python test.py 'data/tokenizer.h5' 'experiments/bi_lstm_large/weights/epoch_18.h5' \
--model_name 'bi_lstm_large' \
--test_data_file 'data/test.notone' \
--ground_truth_file 'data/test.tone' \
--cuda