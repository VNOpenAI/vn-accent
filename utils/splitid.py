import os
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('test_file')
    parser.add_argument('--id', default='id.txt')
    parser.add_argument('--sent', default='sent.txt')
    args = parser.parse_args()

    file_path = args.test_file
    file_dir = os.path.dirname(file_path)
    id_file_path = os.path.join(file_dir, args.id)
    sent_file_path = os.path.join(file_dir, args.sent)

    with open(file_path) as f:
        data = f.read().split('\n')[:-1]
    
    with open(id_file_path, 'w') as id_f, open(sent_file_path, 'w') as sent_f:
        for s in data:
            ix = s[:3]
            sent = s[4:]
            id_f.write(ix+'\n')
            sent_f.write(sent+'\n')
