import os

from argparse import Namespace
from collections import Counter
import nltk.data
import numpy as np
import pandas as pd
import re
import json
from tqdm import tqdm
from konlpy.tag import Okt

# Global vars
MASK_TOKEN = "<MASK>"

args = Namespace(
    raw_dataset_txt="data/modu_txt/NIKL_NEWSPAPER_2022_v1.0/NIRW2200000001.txt",
    window_size=2,
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="data/books/500_bart_test.csv",
    seed=1337,
)

# Split the raw text book into sentences
with open(args.raw_dataset_txt, "r", encoding="utf-8") as file:
    sentences = file.readlines()
sentences = sentences[:500]

okt = Okt()
tokenized_list = list()
vocab = {MASK_TOKEN: 0}
vocab_idx = 0
for i in tqdm(range(len(sentences))):
    if sentences[i][0] == " ":
        sentences[i] = sentences[i][1:]
    tokenized_document = okt.pos(sentences[i])
    temp = list()
    for token in tokenized_document:
        if token[1].lower() not in ["josa", "punctuation", "foreign"]:
            temp.append(token[0])
            try:
                vocab[token[0]]
            except KeyError:
                vocab[token[0]] = vocab_idx + 1
                vocab_idx += 1
    tokenized_list.append(temp)


print(len(sentences), "sentences")
print("Sample:", sentences[100])

okt = Okt()
tokenized_list = list()
vocab = {MASK_TOKEN: 0}
vocab_idx = 0
for i in tqdm(range(len(sentences))):
    if sentences[i][0] == " ":
        sentences[i] = sentences[i][1:]
    tokenized_document = okt.pos(sentences[i])
    temp = list()
    for token in tokenized_document:
        if token[1].lower() not in ["josa", "punctuation", "foreign"]:
            temp.append(token[0])
            try:
                vocab[token]
            except KeyError:
                vocab[token] = vocab_idx + 1
                vocab_idx += 1
    tokenized_list.append(temp)


print(len(sentences), "sentences")
print("Sample:", sentences[100])

flatten = lambda outer_list: [item for inner_list in outer_list for item in inner_list]
windows = list()
for sentence in tqdm(tokenized_list):
    ngrams = list(
        nltk.ngrams(
            [MASK_TOKEN] * args.window_size + sentence + [MASK_TOKEN] * args.window_size, args.window_size * 2 + 1
        )
    )
    for data in ngrams:
        windows.append([data, " ".join(sentence)])

# Create cbow data
data = []
for window in tqdm(windows):
    target_token = window[0][args.window_size]
    context = []
    for i, token in enumerate(window[0]):
        if i == args.window_size:
            continue
        else:
            context.append(token)
    # 대상과 정답
    data.append([target_token, " ".join(token for token in context), window[1]])


# Convert to dataframe
cbow_data = pd.DataFrame(data, columns=["context", "target", "sentence"])

# Create split data
n = len(cbow_data)


def get_split(row_num):
    if row_num <= n * args.train_proportion:
        return "train"
    elif (row_num > n * args.train_proportion) and (row_num <= n * args.train_proportion + n * args.val_proportion):
        return "val"
    else:
        return "test"


cbow_data["split"] = cbow_data.apply(lambda row: get_split(row.name), axis=1)

print(cbow_data.head())

with open("raw_data/500_fasttext_vocab.json", "w") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)

train_cnt = Counter()
eval_cnt = Counter()
test_cnt = Counter()
unique_train = set()
unique_eval = set()
unique_test = set()
for i, series in cbow_data.iterrows():
    if series["split"] == "train":
        unique_train.add(series["sentence"])
    elif series["split"] == "val":
        unique_eval.add(series["sentence"])
    else:
        unique_test.add(series["sentence"])

for item in unique_train:
    train_cnt.update(item.split())
for item in unique_eval:
    eval_cnt.update(item.split())
for item in unique_test:
    test_cnt.update(item.split())

with open("raw_data/500_train_cnt.json", "w") as f:
    json.dump(dict(train_cnt), f, ensure_ascii=False, indent=4)
with open("raw_data/500_eval_cnt.json", "w") as f:
    json.dump(dict(eval_cnt), f, ensure_ascii=False, indent=4)
with open("raw_data/500_test_cnt.json", "w") as f:
    json.dump(dict(test_cnt), f, ensure_ascii=False, indent=4)

cbow_data.pop("sentence")
cbow_data.to_csv(args.output_munged_csv, index=False)
