python test.py 'data/tokenizer.h5' 'experiments/transformer_evolved/weights/epoch_14.h5' \
--model_name 'transformer_evolved' \
--test_data_file 'data/test.notone' \
--ground_truth_file 'data/test.tone' \
--cuda