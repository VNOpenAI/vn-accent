import os
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='id.txt')
    parser.add_argument('--sent', default='sent.txt')
    parser.add_argument('--output', default='output.txt')
    args = parser.parse_args()

    id_file_path = args.id
    sent_file_path = args.sent
    output_file = args.output

    with open(id_file_path) as f:
        list_id = f.read().split('\n')[:-1]
    
    with open(sent_file_path) as f:
        list_sent = f.read().split('\n')[:-1]

    with open(output_file, 'w') as f:
        for i,s in zip(list_id, list_sent):
            f.write(i+','+s+'\n')
