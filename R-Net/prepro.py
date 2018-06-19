# coding=utf-8
import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
import os.path

nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    """
    读取训练集和开发集数据，将数据转换成字典格式。其中token为单词，char为字母
    :param filename: 
    :param data_type: 
    :param word_counter: 
    :param char_counter: 
    :return: 
    """
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        # 原始数据是一个json数据。有一个'data'包含了所有的数据，是一个list
        for article in tqdm(source["data"]):
            # 每篇文章有多个段落。每个段落有若干问题和答案。每个问题的答案可能出现在段落多个地方
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                # 将一个段落转换成单词列表
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                # 找到每个单词在原文字符串的位置
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    # 每个单词的频率等于每段中问题的个数。例如一个段落中只有2个问题，那么频率为2
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        # 字符类似于单词
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    # 一个问题下的所有答案
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        # 这里为什么不直接把answer_start and answer_end加入y1s， y2s？
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        # 找到所有答案的位置
                        y1s.append(y1)
                        y2s.append(y2)
                    # 么个问题一个example
                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None, token2idx_dict=None):
    """
    从glove中读取预训练的embedding，或者随机初始化embedding
    :param counter: 
    :param data_type: 
    :param limit: 
    :param emb_file: 
    :param size: 
    :param vec_size: 
    :param token2idx_dict: 
    :return: 
    """
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.01) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(
        embedding_dict.keys(), 2)} if token2idx_dict is None else token2idx_dict # {'单词': 单词索引}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]  # NULL或者OOV词向量为0
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}  # 与token2idx_dict中的idx一致
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]  # 文章中所有词的矩阵
    return emb_mat, token2idx_dict


def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
    """转换数据，并将处理结果写到out_file的tfRecord文件中
    :param config:
    :param examples: 由process——file生成
    :param data_type:
    :param out_file:
    :param word2idx_dict:
    :param char2idx_dict:
    :param is_test:
    :return:
    """

    # 长度限制
    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    char_limit = config.char_limit

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1
        # 丢弃超长句子
        if filter_func(example, is_test):
            continue

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)  # 将文章转换成idx，其他位置补0

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)
        # 最后都取的最后一个作为答案的位置
        start, end = example["y1s"][-1], example["y2s"][-1]
        y1[start], y2[end] = 1.0, 1.0   # 将答案的起始位置标为1，其他为0
        # 写入文件的结果都是字符的索引
        record = tf.train.Example(features=tf.train.Features(feature={
                                  "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                                  "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                                  "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                                  "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                                  "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                                  "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                                  "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
                                  }))
        writer.write(record.SerializeToString())
    print("Build {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro(config):
    word_counter, char_counter = Counter(), Counter()
    # config.train_file = os.path.join(home, "data", "squad", "train-v1.1.json")
    train_examples, train_eval = process_file(
        config.train_file, "train", word_counter, char_counter)

    # config.dev_file = os.path.join(home, "data", "squad", "dev-v1.1.json")
    dev_examples, dev_eval = process_file(
        config.dev_file, "dev", word_counter, char_counter)

    # os.path.join(home, "data", "squad", "dev-v1.1.json")
    test_examples, test_eval = process_file(
        config.test_file, "test", word_counter, char_counter)

    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim   # config.char_dim = 8

    word2idx_dict = None
    # word2idx_file = os.path.join(target_dir, "word2idx.json")
    if os.path.isfile(config.word2idx_file):
        with open(config.word2idx_file, "r") as fh:
            word2idx_dict = json.load(fh)
    # word_emb_mat 来源于glove
    word_emb_mat, word2idx_dict = get_embedding(word_counter, "word", emb_file=word_emb_file,
                                                size=config.glove_word_size, vec_size=config.glove_dim, token2idx_dict=word2idx_dict)

    char2idx_dict = None
    if os.path.isfile(config.char2idx_file):
        with open(config.char2idx_file, "r") as fh:
            char2idx_dict = json.load(fh)
    # char_embeding 随机生成的
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim, token2idx_dict=char2idx_dict)
    # word2idx一般都是有get_embediing那一步生成的
    build_features(config, train_examples, "train",
                   config.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev",
                              config.dev_record_file, word2idx_dict, char2idx_dict)
    test_meta = build_features(config, test_examples, "test",
                               config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)

    # 写入了word2idx， char2idx， word_emb, char_emb几个数据比较有用
    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.word2idx_file, word2idx_dict, message="word2idx")
    save(config.char2idx_file, char2idx_dict, message="char2idx")
    save(config.test_meta, test_meta, message="test meta")
