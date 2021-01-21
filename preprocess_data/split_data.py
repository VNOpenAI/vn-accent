from accent_utils import *
import random


data = []

with open("data/wikipedia.txt", 'r', encoding='utf-8') as f:
    wiki_data = f.read().split("\n")
    print("Wiki: {} sentences".format(len(wiki_data)))
    data += wiki_data

with open("data/yhoc.txt", 'r', encoding='utf-8') as f:
    yhoc_data = f.read().split("\n")
    print("YHoc: {} sentences".format(len(yhoc_data)))
    data += yhoc_data

random.seed(42)
random.shuffle(data)

X = data
y = [remove_tone_line(x) for x in X]

X_val = X[:10000]
y_val = y[:10000]
X_test = X[10000:20000]
y_test = y[10000:20000]
X_train = X[20000:]
y_train = y[20000:]

with open("data/train.tone", 'w', encoding='utf-8') as f:
    f.write("\n".join(X_train))
with open("data/train.notone", 'w', encoding='utf-8') as f:
    f.write("\n".join(y_train))
print("Train: {} samples".format(len(X_train)))

with open("data/val.tone", 'w', encoding='utf-8') as f:
    f.write("\n".join(X_val))
with open("data/val.notone", 'w', encoding='utf-8') as f:
    f.write("\n".join(y_val))
print("Val: {} samples".format(len(X_val)))

with open("data/test.tone", 'w', encoding='utf-8') as f:
    f.write("\n".join(X_test))
with open("data/test.notone", 'w', encoding='utf-8') as f:
    f.write("\n".join(y_test))
print("Test: {} samples".format(len(X_test)))
