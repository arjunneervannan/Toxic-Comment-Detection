# -*- coding: utf-8 -*-
import re
import string
import os
import numpy as np
import pandas as pd
import spacy
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from spacy.tokenizer import Tokenizer

prefix_re = re.compile(r'''^[\[\("']''')
suffix_re = re.compile(r'''[\]\)"']$''')
# infix_re = re.compile(r'''[\‘\’\`\“\”\"~]-''')
infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')


def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                     suffix_search=suffix_re.search,
                     infix_finditer=infix_re.finditer,
                     token_match=None)


tqdm.pandas()
nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = custom_tokenizer(nlp)
nlp.add_pipe(nlp.create_pipe('sentencizer'))


def clean_data(str_a):
    str_a = re.sub(r'[^\x00-\x7f]', r'', str_a)
    str_a = re.sub(r"\'s", " is", str_a)
    str_a = re.sub(r"\'m", " am", str_a)
    str_a = re.sub(r"\'ve", " have", str_a)
    str_a = re.sub(r"\'ll", " will", str_a)
    str_a = re.sub(r"n\'t", " not", str_a)
    str_a = re.sub(r"\'d", " would", str_a)
    str_a = re.sub(r"\'nt", " not", str_a)
    str_a = re.sub(r"\'re", " are", str_a)
    return str_a.lower()


def load_glove_embedding(path, dim, word_index):
    embeddings_index = {}
    f = open(path)

    print('Generating GloVe embedding...')
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
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print('Loaded GloVe embedding')
    return embedding_matrix


def tag(text):
    flattened = [val for sublist in text for val in sublist]
    my_lst_str = ' '.join(map(str, flattened))

    nlp1 = spacy.load('en')
    doc = nlp1(unicode(my_lst_str, "utf-8"))
    x = [(t.text, t.pos_) for t in doc]
    return x


def normalize(text):
    text = text.lower().strip()
    nlp.tokenizer = custom_tokenizer(nlp)
    doc = nlp(str.encode(text).decode("utf-8", 'ignore')) #TODO
    filtered_sentences = []
    for sentence in doc.sents:
        filtered_tokens = list()
        for i, w in enumerate(sentence):
            s = w.string.strip()
            # if len(s) == 0 or s in string.punctuation and i < len(doc) - 1:
            #     continue
            filtered_tokens.append(s)
        filtered_sentences.append(' '.join(filtered_tokens))
    return filtered_sentences


def process_sent(text):
    doc = nlp(str.encode(text).decode("utf-8", 'ignore')) #TODO
    filtered_sentences = []
    for sentence in doc.sents:
        filtered_tokens = list()
        for i, w in enumerate(sentence):
            s = w.string.strip()
            # if len(s) == 0 or s in string.punctuation and i < len(doc) - 1:
            #     continue
            filtered_tokens.append(s)
        filtered_sentences.append(' '.join(filtered_tokens))
    return filtered_sentences


def chunk_to_arrays(chunk):
    x = chunk['text_tokens'].values
    y = chunk['is_toxic'].values
    return x, y


def load_normalized(x_input):
    x_input['text_tokens'] = x_input['comment'].map(lambda x: normalize(x))
    test_set = x_input.copy()
    test_set['len'] = test_set['text_tokens'].apply(lambda x: len(x))
    test_set = test_set[test_set['len'] != 0]

    return test_set['text_tokens'].values


def load_testdata(test_data_path="data/wiki_test.csv"):
    print('loading test dataset...')
    data_test = pd.read_csv(test_data_path, usecols=['comment', 'is_toxic']).dropna()

    data_test['text_tokens'] = data_test['comment'].progress_apply(lambda x: normalize(x))
    test_set = data_test.copy()
    test_set['len'] = test_set['text_tokens'].apply(lambda x: len(x))
    test_set = test_set[test_set['len'] != 0]
    test_x, test_y = chunk_to_arrays(test_set)
    test_y = to_categorical(test_y)
    print('finished loading data')
    del test_set

    return data_test, test_x, test_y


def load_validdata(valid_data_path="data/wiki_dev.csv"):
    print('loading validation dataset...')
    data_valid = pd.read_csv(valid_data_path, usecols=['comment', 'is_toxic']).dropna()

    data_valid['text_tokens'] = data_valid['comment'].progress_apply(lambda x: normalize(x))
    valid_set = data_valid.copy()
    valid_set['len'] = valid_set['text_tokens'].apply(lambda x: len(x))
    valid_set = valid_set[valid_set['len'] != 0]
    valid_x, valid_y = chunk_to_arrays(valid_set)
    valid_y = to_categorical(valid_y)
    print('finished loading data')
    del (valid_set)

    return data_valid, valid_x, valid_y


def load_traindata(train_data_path="data/wiki_train.csv", model_type='baseline', sent_aug=0):
    print('loading train dataset...')
    data_train = pd.read_csv(train_data_path, usecols=['comment', 'is_toxic']).dropna()
    data_train['text_tokens'] = data_train['comment'].progress_apply(lambda x: normalize(x))
    train_set = data_train.copy()

    if model_type == 'debiased':
        filename = os.path.join('debiased_' + str(sent_aug), 'debiased_' + str(sent_aug) + '.csv')
        print("Augmenting with debiased dataset")
        neutral_train = pd.read_csv(filename, usecols=['comment', 'is_toxic']).dropna()
        neutral_train['text_tokens'] = neutral_train['comment'].progress_apply(lambda x: normalize(x))
        train_set = train_set.append(neutral_train)
        # Shuffle dataset after adding neutral comments
        train_set = train_set.sample(frac=1).reset_index(drop=True)

    if model_type == 'control':
        filename = os.path.join('control_' + str(sent_aug), 'control_' + str(sent_aug) + '.csv')
        print("Augmenting with control dataset")
        control_train = pd.read_csv(filename, usecols=['comment', 'is_toxic']).dropna()
        control_train['text_tokens'] = control_train['comment'].progress_apply(lambda x: normalize(x))
        train_set = train_set.append(control_train)
        # Shuffle dataset after adding neutral comments
        train_set = train_set.sample(frac=1).reset_index(drop=True)

    train_set['len'] = train_set['text_tokens'].apply(lambda x: len(x))
    train_set = train_set[train_set['len'] != 0]
    train_x, train_y = chunk_to_arrays(train_set)

    train_y = to_categorical(train_y)
    print('finished loading train data')
    del (train_set, data_train)

    return train_x, train_y


def load_data(model_type, sent_aug):
    train_x, train_y = load_traindata(model_type=model_type, sent_aug=sent_aug)
    m, valid_x, valid_y = load_validdata()
    m1, test_x, test_y = load_testdata()
    print(train_x.shape)
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def noun_and_adj():
    print('loading train, validation and test datasets...')
    training_data_path = "data/wiki_train.csv"
    valid_data_path = "data/wiki_dev.csv"

    data_train = pd.read_csv(training_data_path, usecols=['comment', 'is_toxic']).dropna()
    data_valid = pd.read_csv(valid_data_path, usecols=['comment', 'is_toxic']).dropna()

    data_train['comment'] = data_train['comment'].map(lambda x: clean_data(x))
    data_valid['comment'] = data_valid['comment'].map(lambda x: clean_data(x))

    nlp1 = spacy.load('en')
    noun = []
    adj = []
    # verb = []
    step = 1000
    print("Finding nouns that are used as adjectives... ")
    for idx in range(0, len(data_train), step):
        document = ". ".join(data_train['comment'][idx:idx + step - 1])
        doc = nlp1(unicode(document, "utf-8"), disable=['parser', 'ner'])
        noun.append([t.text for t in doc if (t.pos_ == "NOUN") | (t.pos_ == "PROPN")])
        adj.append([t.text for t in doc if t.pos_ == "ADJ"])
        # verb.append([t.text for t in doc if t.pos_ == "VERB"])
    print("processed train set")

    for idx in range(0, len(data_valid), step):
        document = ". ".join(data_valid['comment'][idx:idx + step - 1])
        doc = nlp1(unicode(document, "utf-8"), disable=['parser', 'ner'])
        noun.append([t.text for t in doc if t.pos_ == "NOUN"])
        noun.append([t.text for t in doc if t.pos_ == "PROPN"])
        adj.append([t.text for t in doc if t.pos_ == "ADJ"])
        # verb.append([t.text for t in doc if t.pos_ == "VERB"])
    print("processed validation set")

    flattened_nouns = sorted(set([item for sublist in noun for item in sublist]))
    nouns = [x.encode('utf-8') for x in flattened_nouns]
    flattened_adjectives = sorted(set([item for sublist in adj for item in sublist]))
    adjectives = [x.encode('utf-8') for x in flattened_adjectives]
    # flattened_verbs = sorted(set([item for sublist in verb for item in sublist]))
    # verbs = [x.encode('utf-8') for x in flattened_verbs]

    my_list = list(sorted(set(nouns) - (set(nouns) - set(adjectives))))
    # my_list12 = list(set(my_list) - set(verbs))
    my_list23 = [item for item in my_list if len(item) >= 3]
    noun_adj = [item for item in my_list23 if item.isalpha()]
    df_nounadj = pd.DataFrame(noun_adj)
    df_nounadj.columns = ['term']
    print("extracted nouns and adjs...")

    df_idf = tfidf(data_train, data_valid)
    print("IDF calculated for entire corpus...")

    s1 = pd.merge(df_nounadj, df_idf, how='inner', on=['term'])

    # identity_terms = [x.encode('utf-8') for x in nounadj_idf.term.values]

    with open('data/words_alpha.txt', 'r') as f:
        words = f.read().split()
    df_words = pd.DataFrame(words, columns=['term'])
    s2 = pd.merge(s1, df_words, how='inner', on=['term'])
    s3 = s2[s2['term'].apply(lambda x: len(x) >= 3)]
    s4 = s3[s3['term'].apply(lambda x: x.isalpha())]

    print("Writing IDF for nouns that are used as adjectives to a file")
    writer = pd.ExcelWriter('idf_noun_adj.xlsx')
    df_nounadj.to_excel(writer, 'noun_adj', index=False)
    df_idf.to_excel(writer, 'idf_all_terms', index=False)
    s1.to_excel(writer, 'IDF_for_noun_adj', index=False)
    s4.to_excel(writer, 'filtered_english', index=False)
    writer.save()


def tfidf(data_train, data_valid):
    all_texts = []
    for idx in range(data_train['comment'].shape[0]):
        text = data_train['comment'][idx]
        all_texts.append(text)
    for idx in range(data_valid['comment'].shape[0]):
        all_texts.append(data_valid['comment'][idx])

    # If ``smooth_idf=True`` (the default), the constant "1" is added to the
    # numerator and denominator of the idf as if an extra document was seen
    # containing every term in the collection exactly once, which prevents
    # zero divisions: idf(d, t) = log [ (1 + n) / 1 + df(d, t) ] + 1.
    print("Computing IDF...")
    vectorizer = TfidfVectorizer(use_idf=True, norm=None, smooth_idf=False, sublinear_tf=False, binary=False, min_df=1,
                                 max_df=1.0, max_features=None, strip_accents='unicode', ngram_range=(1, 1),
                                 preprocessor=None, stop_words='english')
    X = vectorizer.fit_transform(all_texts)
    idf = vectorizer.idf_ - 1

    idf_map = dict(zip(vectorizer.get_feature_names(), idf))
    idf_df = pd.DataFrame.from_dict(idf_map.items(), orient='columns')
    idf_df.columns = ['term', 'IDF']

    df = pd.DataFrame(idf_df.sort_values(by='IDF', ascending=False))
    df.columns = ['term', 'IDF']
    df = df[~df.term.str.contains(r'[0-9]')]
    df = df[df['term'].str.isalpha()]
    return df
