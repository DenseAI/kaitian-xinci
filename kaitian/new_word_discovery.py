# -*- coding:utf-8 -*-

import os
import re
import math
import time
import json
import pandas as pd
import codecs
import numpy as np
import logging

from kaitian.transformer.transformer_new_word_candidate import get_or_create

def is_not_chinese(uchar:str):
    """
    判断一个unicode是否是汉字
    Determine if a unicode is a Chinese character
    """
    if uchar.isalpha() is True:
        return False
    elif uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return False
    else:
        return True


def extract_candidate_word(_doc, _max_word_len):
    """
    提取候选词
    Extracting candidate words
    :param _doc:
    :param _max_word_len:
    :return:
    """
    candidates = []
    doc_length = len(_doc)
    for ii in range(doc_length):
        for jj in range(ii + 1, min(ii + 1 + _max_word_len, doc_length + 1)):

            # 判断是否是中文字符，非中文字符，不再组合
            # Determine whether it is Chinese characters, non-Chinese characters, no longer combined
            if is_not_chinese(doc[ii:jj]) is True or is_not_chinese(doc[jj - 1:jj]) is True:
                break

            word = doc[ii:jj]
            previous_word = '~'
            if ii - 1 >= 0:
                previous_word = doc[ii - 1:ii]
            next_word = '~'
            if jj + 1 < doc_length + 1:
                next_word = doc[jj:jj + 1]

            # TODO: 存在性能问题，大型文档例如Wikipedia dumps等，几个G的文档可能内存顶不住
            #      There are performance issues, large documents such as Wikipedia dumps, etc.
            #      Several G documents may not be able to load to memory
            candidates.append([previous_word, word, next_word])
    return candidates


def gen_bigram(_word_str):
    """
    一个单词拆分为所有可能的两两组合。例如，ABB可以分为（a，bb），（ab，b）。
    A word is divide into two part by following all possible combines.
    For instance, ABB can divide into (a,bb),(ab,b)
    :param _word_str:
    :return:
    """
    return [(_word_str[0:_i], _word_str[_i:]) for _i in range(1, len(_word_str))]


def compute_entropy(_list):
    """
    计算熵  https://zh.wikipedia.org/zh-hans/%E7%86%B5_(%E4%BF%A1%E6%81%AF%E8%AE%BA)
           https://baike.baidu.com/item/熵/19190273

    Calculating entropy
    :param _list:
    :return:
    """
    length = float(len(_list))
    frequence = {}
    if length == 0:
        return 0, frequence
    else:
        for i in _list:
            frequence[i] = frequence.get(i, 0) + 1
        return sum(map(lambda x: - x / length * math.log(x / length), frequence.values())), frequence


class WordInfo(object):
    """
    记录每个候选单词信息，包括左邻居，右邻居，频率，PMI
    Record every candidate word information include left neighbors, right neighbors, frequency, PMI
    """

    def __init__(self, text):
        super(WordInfo, self).__init__()
        self.text = text
        self.freq = 0.0
        self.left = []         # record left neighbors
        self.left_dict = {}
        self.right = []        # record right neighbors
        self.right_dict = {}
        self.pmi = 0

        self.raw_freq = 0
        self.raw_length = 0

    def update_data(self, left, right):
        self.freq += 1.0
        if left:
            self.left.append(left)
        if right:
            self.right.append(right)

    def compute_indexes(self, length):
        # 计算单词的频率和左/右熵
        # compute frequency of word,and left/right entropy
        self.raw_freq = self.freq
        self.raw_length = length
        self.freq /= length
        self.left, self.left_dict = compute_entropy(self.left)
        self.right, self.right_dict = compute_entropy(self.right)

    def compute_pmi(self, words_dict):
        # 计算单词的各种组合
        # compute all kinds of combines for word
        sub_part = gen_bigram(self.text)
        if len(sub_part) > 0:
            self.pmi = min(
                map(lambda word: math.log(self.freq / words_dict[word[0]].freq / words_dict[word[1]].freq), sub_part))


class DocumentSegment(object):
    """
    Main class for Chinese word discovery

    reference:
    1. http://www.matrix67.com/blog/archives/5044
    """

    def __init__(self, doc, max_word_len=5, min_tf=0.000005, min_freq=5, min_entropy=0.05, min_pmi=3.0):
        super(DocumentSegment, self).__init__()
        self.max_word_len = max_word_len
        self.min_tf = min_tf
        self.min_freq = min_freq
        self.min_entropy = min_entropy
        self.min_pmi = min_pmi

        # analysis documents
        self.word_info = self.gen_words(doc)

        count = float(len(self.word_info))
        self.avg_frq = sum(map(lambda w: w.freq, self.word_info)) / count
        self.avg_entropy = sum(map(lambda w: min(w.left, w.right), self.word_info)) / count
        self.avg_pmi = sum(map(lambda w: w.pmi, self.word_info)) / count

        filter_function = lambda f: len(f.text) > 1 and f.pmi > self.min_pmi and f.freq > self.min_tf \
                                    and min(f.left, f.right) > self.min_entropy and f.raw_freq > self.min_freq
        self.word_tf_pmi_ent = map(lambda w: (w.text, len(w.text), w.freq, w.pmi, min(w.left, w.right)),
                                   filter(filter_function, self.word_info))

    def gen_words(self, doc):

        # 过滤掉非中文字符
        # Filter out non-Chinese characters
        pattern = re.compile(u'[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=+，。《》、？：；“”‘’｛｝【】（）…￥！—┄－]+')
        doc = pattern.sub(r' ', doc)

        word_index = extract_candidate_word(doc, self.max_word_len)
        word_cad = {}  # 后选词的字典
        for suffix in word_index:
            word = suffix[1]
            previous_word = suffix[0]
            next_word = suffix[2]
            if word not in word_cad:
                word_cad[word] = WordInfo(word)

            # 记录候选词的左右邻居
            # record frequency of word and left neighbors and right neighbors
            word_cad[word].update_data(previous_word, next_word)
        length = len(doc)

        # 计算候选词的频率、以及左右熵
        # computing frequency of candidate word and entropy of left/right neighbors
        for word in word_cad:
            word_cad[word].compute_indexes(length)

        # ranking by length of word
        values = sorted(word_cad.values(), key=lambda x: len(x.text))
        for v in values:
            if len(v.text) == 1:
                continue
            v.compute_pmi(word_cad)
        # ranking by freq
        return sorted(values, key=lambda v: len(v.text), reverse=False)


def load_words(filename):
    max_len = 0
    f = open(filename, encoding='utf-8')
    objects = json.load(f)
    words = []
    if objects is not None and len(objects) > 0:
        for word in objects:
            words.append(word['more'])
            if(len(word['word'])) > max_len:
                max_len = len(word['word'])
    return words, max_len



def load_dictionary(config_path, encoding="utf-8"):
    """
    Load dict
    :param config_path:
    :param encoding:
    :return:
    """

    with open(config_path, mode="r", encoding=encoding) as file:
        str = file.read()
        config = json.loads(str)
        return config


def load_model(model_name, show_summary=False):
    """
    加载模型
    """
    config_save_path = "./model/{}_default_config.json".format(model_name)  # config path
    source_dict_path = "./model/{}_source_dict.json".format(model_name)     # 源字典路径
    target_dict_path = "./model/{}_target_dict.json".format(model_name)     # 目标字典路径
    word_dict_path = "./model/{}_word_dict.json".format(model_name)         # 源字典路径
    label_dict_path = "./model/{}_label_dict.json".format(model_name)       # 目标字典路径
    model_path = "./model/{}_weights.hdf5".format(model_name)               # 模型路径

    #加载标签字典
    word_dict = load_dictionary(word_dict_path)
    transformer_model = get_or_create(config_save_path,
                              src_dict_path=source_dict_path,
                              tgt_dict_path=target_dict_path,
                              weights_path=model_path)
    if show_summary is True:
        transformer_model.model.summary()

    return transformer_model, word_dict


def new_word_candidate_score(texts:list, transformer_model, word_dict:dict):
    """
    对候选词进行评分
    :param texts:
    :param transformer_model:
    :param word_dict:
    :return:
    """
    pred_texts = []
    pred_sequences = []

    for text in texts:
        text = text.lower()
        CLS = '[CLS]'

        words = []
        words.append(word_dict[CLS])
        for ii in range(len(text)):
            x = word_dict.get(text[ii].lower(), 2)  # 2 <UNK>
            words.append(x)

        pred_sequences.append(words)
        pred_texts.append(text)

    results = transformer_model.decode_texts(pred_texts, pred_sequences)
    scores = []
    for text, result in zip(pred_texts, results):
        scores.append(result[0])

    return scores

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    starttime = time.clock()

    path = os.path.abspath('.')
    word_candidate = []

    # 加载字典
    dict_bank = []
    dict_path = path + '/dict/dict.txt'
    # for i in codecs.open(dict_path, 'r', "utf-8"):
    #     dict_bank.append(i.split(' ')[0])
    logging.debug("Loading dictionary.")

    # 加载文档
    document_file = '/data/西游记.txt'
    doc = codecs.open(path + document_file, "r", "utf-8").read()
    logging.debug("Loading document.")

    # 加载评分文档
    transformer_model, word_dict = load_model("new_word_all_transformer")
    logging.debug("Loading transformer model.")

    logging.debug("Document analysis:")

    # 统计候选词，并计算熵、PMI
    word = DocumentSegment(doc, max_word_len=5, min_tf=(0.00005), min_entropy=1.0, min_pmi=3.0)
    logging.debug('avg_frq:' + str(word.avg_frq))
    logging.debug('avg_pmi:' + str(word.avg_pmi))
    logging.debug('avg_entropy:' + str(word.avg_entropy))

    logging.debug("New word candidate scoring:")

    # 计算候选词评分
    texts = []
    words = []
    for ii in word.word_tf_pmi_ent:
        texts.append(ii[0])
        words.append(ii)
    scores = new_word_candidate_score(texts, transformer_model, word_dict)
    scores = np.array(scores)


    index = 0
    wordlist = []
    for ii in words:
        if ii[0] not in dict_bank:
            word_candidate.append(ii[0])
            wordlist.append([ii[0], ii[1], ii[2], ii[3], ii[4], scores[index]])
        index += 1

    # ranking on score(primary key), entropy(secondary key) and pmi
    wordlist = sorted(wordlist, key=lambda word: word[3], reverse=True)
    wordlist = sorted(wordlist, key=lambda word: word[4], reverse=True)
    wordlist = sorted(wordlist, key=lambda word: word[5], reverse=True)


    logging.debug("Save new words.")

    seg = pd.DataFrame(wordlist, columns=['word', 'len', 'frequency', 'pmi', 'entropy', 'score'])
    seg.to_csv(path + '/output/extractword.csv', index=False, encoding="utf-8")

    # intersection = set(word_candidate) & set(dict_bank)
    # newwordset = set(word_candidate) - intersection

    endtime = time.clock()
    logging.debug('Times: ' + str(endtime - starttime))
