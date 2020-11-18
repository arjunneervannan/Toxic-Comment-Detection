from __future__ import unicode_literals

import re
import string
import os

import numpy as np
import pandas as pd
import spacy
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
from tqdm import tqdm

tqdm.pandas()
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))


def clean_data(str):
    str = re.sub(r'[^\x00-\x7f]', r'', str)
    str = re.sub(r"\'nt", " not", str)
    str = re.sub(r"\'m", " am", str)
    str = re.sub(r"\'ll", " will", str)
    str = re.sub(r"\'d", " would", str)
    str = re.sub(r"\'ve", " have", str)
    str = re.sub(r"n\'t", " not", str)
    str = re.sub(r"\'re", " are", str)
    str = re.sub(r"\'s", " is", str)
    str = re.sub(r"\s{2,}", " ", str)
    return str.lower()


def tag(text):
    flattened = [val for sublist in text for val in sublist]
    my_lst_str = ' '.join(map(str, flattened))

    nlp1 = spacy.load('en')
    doc = nlp1(unicode(my_lst_str, "utf-8"))
    # x = [(t.text, t.pos_) for t in doc if t.pos_ != 'PUNCT']
    x = [(t.text, t.pos_) for t in doc]
    return x


def fit_tokenizer(texts):
    tokenizer = Tokenizer(filters='"()*,-/;[\]^_`{|}~')
    alltext = []
    for text in texts:
        for sentence in text:
            alltext.append(sentence)

    tokenizer.fit_on_texts(alltext)
    print("------------------------")
    print("Tokenizer fitted!")
    return tokenizer


def load_glove_embedding(path, dim, word_index):
    print("Loading GloVe embedding")
    embeddings_index = {}
    f = open(path)

    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(word_index) + 1, dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def array_fit(text):
    text = text.lower().strip()
    doc = nlp(text.decode('utf8'))
    arrayed_sentence = []

    for sentence in doc.sents:
        filtered_sentence = []
        for i, w in enumerate(sentence):
            s = w.string.strip()
            if len(s) == 0 or s in string.punctuation and i < len(doc) - 1:
                continue
            s = s.replace(',', '.')
            filtered_sentence.append(s)
        arrayed_sentence.append(' '.join(filtered_sentence))
    return arrayed_sentence


def load_normalized(x_input):
    x_input['text_tokens'] = x_input['comment'].map(lambda x: array_fit(x))
    test_set = x_input.copy()
    test_set['len'] = test_set['text_tokens'].apply(lambda x: len(x))
    test_set = test_set[test_set['len'] != 0]

    return test_set['text_tokens'].values


def load_data(which_dataset, saving_path=None, model_type='baseline', num_comments=0):
    path = "data/wiki_" + which_dataset + ".csv"
    text_column = 'comment'
    label_column = 'is_toxic'

    data = pd.read_csv(path, usecols=[text_column, label_column]).dropna()

    if model_type == 'debiased':
        debiased_path = os.path.join(saving_path, 'debiased_' + str(num_comments) + '.csv')
        debiased_data = pd.read_csv(debiased_path, usecols=[text_column, label_column]).dropna()
        data = data.append(debiased_data)
        data = data.sample(frac=1).reset_index(drop=True)

    if model_type == 'control':
        control_path = os.path.join(saving_path, 'control_' + str(num_comments) + '.csv')
        control_data = pd.read_csv(control_path, usecols=[text_column, label_column]).dropna()
        data = data.append(control_data)
        data = data.sample(frac=1).reset_index(drop=True)

    data['text_tokens'] = data[text_column].progress_apply(lambda x: array_fit(x))
    array_data = data.copy()
    array_data['length'] = data['text_tokens'].apply(lambda x: len(x))

    array_data = array_data[array_data['length'] != 0]

    text_x = array_data['text_tokens'].values
    labels_y = to_categorical(array_data['is_toxic'].values)
    del (array_data, data)

    return text_x, labels_y


def noun_adj():
    nlp = spacy.load('en')
    noun = []
    adj = []
    step = 1000

    train, _ = load_data("train")
    dev, _ = load_data("dev")
    alldata = np.append(train, dev)

    for i in range(0, len(train), step):
        subset = train[i: i + step - 1]
        for comment in subset:
            for sentence in comment:
                doc = nlp(sentence, disable=['parser', 'ner'])
                for word in doc:
                    if word.pos_ == "NOUN":
                        noun.append(word.text)
                    if word.pos_ == "ADJ":
                        adj.append(word.text)

    for i in range(0, len(dev), step):
        subset = train[i: i + step - 1]
        for comment in subset:
            for sentence in comment:
                doc = nlp(sentence, disable=['parser', 'ner'])
                for word in doc:
                    if word.pos_ == "NOUN":
                        noun.append(word.text)
                    if word.pos_ == "ADJ":
                        adj.append(word.text)

    noun_adj = list(set(adj).intersection(noun))

    with open('data/words_alpha.txt', 'r') as f:
        words = f.read().split()

    noun_adj_english = pd.merge(pd.DataFrame(noun_adj, columns=['terms']), pd.DataFrame(words, columns=['terms']),
                                how='inner', on=['terms'])
    idf_df = idf(alldata)
    idf_nnadj = pd.merge(idf_df, noun_adj_english, how='inner', on=['terms'])
    writer = pd.ExcelWriter('data/idf_nnadj.xlsx')
    idf_nnadj.to_excel(writer, 'idf_terms', index=False)
    writer.save()
    return idf_nnadj


def idf(dataset):
    allwords = []

    for comment in dataset:
        for sentence in comment:
            allwords.append(sentence)

    vectorizer = TfidfVectorizer(use_idf=True, norm=None, smooth_idf=False, sublinear_tf=False, binary=False, min_df=1,
                                 max_df=1.0, max_features=None, strip_accents='unicode', ngram_range=(1, 1),
                                 preprocessor=None, stop_words='english')
    X = vectorizer.fit_transform(allwords)

    idf_df = pd.DataFrame(zip(vectorizer.get_feature_names(), vectorizer.idf_), columns=['terms', 'IDF'])

    return idf_df


def check(input_data, input_str):
    matches = [c for c in input_data.lower().split() if c in input_str]
    if len(matches) == 0:
        return None
    else:
        return matches
