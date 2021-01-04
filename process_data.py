from accent_utils import *
import random
from sklearn.model_selection import train_test_split


with open("data/wiki.txt", 'r', encoding='utf-8') as f:
    data = f.read().split("\n")

with open("data/yhoc.txt", 'r', encoding='utf-8') as f:
    data += f.read().split("\n")

random.seed(42)
random.shuffle(data)

X = data
y = [remove_tone_line(x) for x in X]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

with open("data/train.tone", 'w', encoding='utf-8') as f:
    f.write("\n".join(X_train))
with open("data/train.notone", 'w', encoding='utf-8') as f:
    f.write("\n".join(y_train))

with open("data/val.tone", 'w', encoding='utf-8') as f:
    f.write("\n".join(X_val))
with open("data/val.notone", 'w', encoding='utf-8') as f:
    f.write("\n".join(y_val))

with open("data/test.tone", 'w', encoding='utf-8') as f:
    f.write("\n".join(X_test))
with open("data/test.notone", 'w', encoding='utf-8') as f:
    f.write("\n".join(y_test))
