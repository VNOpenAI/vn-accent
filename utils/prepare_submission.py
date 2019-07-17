import os
import argparse

from accent import split_word_tone

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='id.txt')
    parser.add_argument('--sent', default='sent.txt')
    parser.add_argument('--output', default='submission.txt')
    args = parser.parse_args()

    id_file_path = args.id
    sent_file_path = args.sent
    output_file = args.output

    with open(id_file_path) as f:
        list_id = f.read().split('\n')[:-1]
    
    with open(sent_file_path) as f:
        list_sent = f.read().split('\n')[:-1]
            sent_f.write(sent+'\n')

    
    with open(output_file, 'w') as f:
        for idx,sent in zip(list_id, list_sent):
            words = sent.split()
            for w in words:
                _, tone_num = split_word_tone(w)
                f.write(idx+','+str(tone_num)+'\n')
