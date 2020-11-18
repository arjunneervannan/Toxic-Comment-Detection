# -*- coding: utf-8 -*-
# !/usr/bin/env python
from __future__ import division
from __future__ import unicode_literals

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from nltk.corpus import stopwords
from numpy.random import seed
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow import set_random_seed
from tqdm import tqdm

import data as data
from model import HierarchicalAttn

seed(10987)
set_random_seed(78901)

stop = stopwords.words('english')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('-model_type', '--model_type', metavar='str', type=str, default='baseline')
parser.add_argument('-sentence_augment', '--sentence_augment', metavar='int', type=int, default=0)
parser.add_argument('-generate_noun_and_adj', '--generate_noun_and_adj', metavar='bool', type=bool, default=False)
parser.add_argument('-regenerate', '--regenerate', metavar='bool', type=bool, default=False)
parser.add_argument('-is_train', '--is_train', metavar='bool', type=bool, default=False)
parser.add_argument('-is_test', '--is_test', metavar='bool', type=bool, default=False)
parser.add_argument('-is_test2', '--is_test2', metavar='bool', type=bool, default=False)
parser.add_argument('-is_validation', '--is_validation', metavar='bool', type=bool, default=False)
parser.add_argument('-single_test', '--single_test', metavar='bool', type=bool, default=False)
parser.add_argument('-a_test', '--a_test', metavar='bool', type=bool, default=False)
parser.add_argument('-cutoff', '--cutoff', metavar='float', type=float, default=1.0)
parser.add_argument('-merge_results', '--merge_results', metavar='bool', type=bool, default=False)
parser.add_argument('-dirname', '--dirname', metavar='str', type=str, default='')
parser.add_argument('-compute_auc', '--compute_auc', metavar='bool', type=bool, default=False)

parser.add_argument('-new_test', '--new_test', metavar='bool', type=bool, default=False)

args = parser.parse_args(sys.argv[1:])

SAVED_MODEL_DIR = args.model_type + '_' + str(args.sentence_augment)
SAVED_MODEL_FILENAME = 'model_' + args.model_type + '.h5'
EMBEDDINGS_PATH = 'glove.6B/glove.840B.300d.txt'
if not os.path.exists(SAVED_MODEL_DIR):
    os.makedirs(SAVED_MODEL_DIR)


def weight2color(brightness):
    """Converts a single (positive) attention weight to a shade of blue."""
    brightness = brightness.item()
    brightness = int(round(255 * brightness))  # convert from 0.0-1.0 to 0-255
    ints = (255, 255 - brightness, 255 - brightness)
    return 'rgba({}, {}, {}, 0.6)'.format(*ints)


def check(input_data, input_str):
    # check if test data comments have the noun_adj combo
    matches = [c for c in input_data.lower().split() if c in input_str]
    if len(matches) == 0:
        return None
    else:
        return matches


def process_classifications(mydata, x, y_predict, y_true, nnadj_list, indices, fname):
    fp_df = pd.DataFrame(mydata.loc[indices])
    fp_df = fp_df.drop(['text_tokens'], axis=1)
    df_fname1 = os.path.join(SAVED_MODEL_DIR, args.model_type + "_falsepositives.csv")

    incorrect_x = x[indices]
    incorr_x = [[x.encode("utf-8") for x in l] for l in incorrect_x]
    incorrect_pred = y_predict[indices]
    incorrect_true = y_true[indices]

    classifcations = pd.DataFrame({'comment': incorr_x, "TrueValue": incorrect_true, "PredictedValue:": incorrect_pred})
    classifcations.to_csv(args.model_type + "_fp.csv")

    sent_parts = list()
    idx2label = {0: 'non-toxic', 1: 'toxic'}
    activation_wts = pd.DataFrame()
    output_text = []
    data_df = pd.DataFrame()

    for idx in tqdm(range(len(incorrect_x))):
        if len(incorrect_x[idx]) < 1:
            continue
        text = '. '.join(map(str, [l.encode('ascii', 'ignore') for l in incorrect_x[idx]]))
        wts, pred = model.get_attn_wts_with_prediction(text, normalized=False)
        data_df = data_df.append({'comment': text, "TrueValue": incorrect_true[idx],
                                  "PredictedValue:": incorrect_pred[idx]}, ignore_index=True)
        flattened_wts = pd.DataFrame(sorted(set([item for sublist in wts for item in sublist])))
        flattened_wts.columns = ['term', 'wts']
        flattened_wts['alter'] = pd.DataFrame(flattened_wts['term'].map(lambda x: check(x, nnadj_list.term.values)))
        flattened_wts = flattened_wts.dropna()
        flattened_wts = flattened_wts.drop(['alter'], axis=1)
        activation_wts = activation_wts.append(flattened_wts, ignore_index=True)

        sent_parts.append('<span style="padding:2px;">[actual: %10s >< pred: %10s]</span> ' % (
            idx2label[incorrect_true[idx]], idx2label[incorrect_pred[idx]]))
        sent_parts.append('<br>')
        for i in range(len(wts)):
            for j in range(len(wts[i])):
                sent_parts.append(
                    '<span style="background: {}; color:#000; padding:2px; font-weight=\'bold\'">{}</span>'.format(
                        weight2color(wts[i][j][1]), wts[i][j][0]))

        sent_parts.append('<br><br><br>')
        output_text = ' '.join(sent_parts)

    outputfilename = os.path.join(SAVED_MODEL_DIR, "output_" + fname + ".html")
    f_out = open(outputfilename, "w")
    f_out.write(output_text)
    f_out.close()
    data_df.to_csv(df_fname1)

    print("Computing min, max and mean attn wts for test data with noun_adj combo")
    activation_wts.columns = ['term', 'wts']
    wts_max = activation_wts.groupby('term', group_keys=False).apply(lambda x: x.loc[x.wts.idxmax()])
    wts_max.columns = ['term', 'wts_max']
    wts_min = activation_wts.groupby('term', group_keys=False).apply(lambda x: x.loc[x.wts.idxmin()])
    wts_min.columns = ['term', 'wts_min']
    wts_mean = activation_wts.groupby('term').mean().reset_index()
    wts_mean.columns = ['term', 'wts_mean']
    wts_mean.set_index('term')
    z1 = pd.merge(wts_max, wts_min, on='term', how='outer')
    z2 = pd.merge(z1, wts_mean, on='term', how='outer')
    attn_wts = pd.merge(z2, nnadj_list, on='term', how='inner')
    attn_wts.columns = ['term', 'max_attnWt', 'min_attnWt', 'mean_attnWt', 'IDF']

    return classifcations, attn_wts


def predict_on(attn_model, mydata, x_input, y_input, nnadj_list, just_test=False):
    encoded_x = attn_model.encode_texts(x_input)
    y_predictprob = attn_model.predict([encoded_x])
    y_predict = np.argmax(y_predictprob, axis=1)
    y_true = np.argmax(y_input.astype(int), axis=1)

    print('ROC AUC:', roc_auc_score(y_true, y_predictprob[:, 1]))
    print('F1 score:', f1_score(y_true, y_predict))
    print('Recall:', recall_score(y_true, y_predict))
    print('Precision:', precision_score(y_true, y_predict))
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
    entireset_fpr = float(fp/(fp+tn))
    print('FPR: ', entireset_fpr)
    print("TN, FP, FN, TP : ", (tn, fp, fn, tp))
    if just_test:
        return entireset_fpr

    # process false positives
    print("\n processing false positives")
    indices_fp = np.where((y_true == 0) & (y_predict == 1))[0]
    class_fp, attn_fp = process_classifications(mydata, x_input, y_predict, y_true, nnadj_list, indices_fp, fname="fp")

    if (args.model_type == 'baseline') and (args.is_train or args.is_validation):
        print("Generating identity terms from valiation set when training baseline model")
        # if args.is_train:
        id_df = attn_fp.filter(['term', 'mean_attnWt', 'IDF'], axis=1)
        id_df = id_df.sort_values(['mean_attnWt', 'IDF'], ascending=[False, False])
        id_df = id_df[id_df['IDF'] > 5.0]

        id_terms = list(id_df.term.values.flatten())
        id_terms = [x.encode('utf-8') for x in id_terms]
        id_terms = [r'\b' + x + r'\b' for x in id_terms]

        # Remove id terms with less than 10 occurences in validation set
        valid = pd.read_csv("data/wiki_dev.csv", usecols=['comment', 'is_toxic']).dropna()
        remove_list = []
        for t in id_terms:
            df_temp = valid[valid['comment'].str.contains(t)]
            if len(df_temp) < 5:
                remove_list.append(t)
        remove_list = [x[2:len(x) - 2] for x in remove_list]
        print ("Removing words: ", remove_list)
        id_df = id_df.set_index("term")
        id_df.drop(remove_list, axis=0, inplace=True)
        id_df = id_df.reset_index()
        id_df1 = id_df[id_df['term'].apply(lambda x: len(x) >= 3)]
        name = 'identity_terms_' + str(args.cutoff) + '.xlsx'
        writer = pd.ExcelWriter(name)
        id_df1.to_excel(writer, sheet_name='filtered', index=False)
        writer.save()

    fwtr = pd.ExcelWriter(os.path.join(SAVED_MODEL_DIR, 'attn_wts_' + args.model_type +
                                       '_' + str(args.sentence_augment) + '.xlsx'))

    attn_fp.to_excel(fwtr, 'fp', index=False)
    fwtr.save()
    return entireset_fpr


def compute_fpr(attn_model, data1, entireset_fpr):
    df = pd.read_excel("identity_terms.xlsx", sheetname='filtered')
    id_terms1 = list(df.term.values.flatten())
    id_terms1 = [x.encode('utf-8') for x in id_terms1]
    id_terms1 = [r'\b' + x + r'\b' for x in id_terms1]

    output_df = pd.DataFrame(columns=['term', 'FPR'])
    for t in id_terms1:
        data1['comment'] = pd.DataFrame(data1['comment'].str.lower())
        temp_df = data1[data1['comment'].str.contains(t)]
        if len(temp_df) == 0:
            print("skipping", t)
            continue

        input_x = data.load_normalized(temp_df)
        encoded_x = attn_model.encode_texts(input_x)
        y_predict = np.argmax(attn_model.predict([encoded_x]), axis=1)
        y_true = list(temp_df['is_toxic'].astype(int))
        returned = confusion_matrix(y_true, y_predict).ravel()

        if len(returned) == 4:
            tn, fp, fn, tp = returned
            if (fp == 0) and (tn == 0):
                fpr = 0.0
            else:
                fpr = float(fp / (fp + tn))
            diff = np.abs(entireset_fpr - fpr)
            output_df = output_df.append({'term': t[2:len(t)-2], 'FPR': fpr, 'Abs Diff from Overall': diff},
                                         ignore_index=True)
    name = args.model_type + '_' + str(args.sentence_augment) + "/" + args.model_type + '_' + str(
        args.sentence_augment) + '_fpr.csv'
    output_df.to_csv(name, index=False)


if __name__ == '__main__':
    # Find nouns that are used as adj in the entire corpus (train, dev and test)
    # and compute IDF for those terms
    if args.generate_noun_and_adj:
        data.noun_and_adj()

    if args.model_type == 'debiased':
        if args.regenerate:
            print("Extracting wikipedia articles to debias training dataset")
            # Load spreadsheet with identity termscp
            df_terms = pd.read_excel("identity_terms.xlsx", sheetname='filtered')
            identity_terms = list(df_terms.term.values.flatten())
            identity_terms = [x.encode('utf-8') for x in identity_terms]
            identity_terms = [r'\b' + x + r'\b' for x in identity_terms]
            # with open('data/wiki_en.txt', 'r') as readfile:
            #     text = readfile.read().decode('utf-8')
            #     text = re.sub(r'[^\x00-\x7f]', r'', text)
            #     sent_tokenize_list = sent_tokenize(text.lower())
            # print ("length of input wiki neutral sentences: ", len(sent_tokenize_list))
            #
            # df_sentences = pd.DataFrame(sent_tokenize_list, columns=['comment'])
            # df_sentences.to_csv("wiki_tokenized.csv")
            df_sentences = pd.read_csv("wiki_tokenized.csv")

            df_sentences['len'] = df_sentences['comment'].apply(lambda x: len(x))
            df_sentences = df_sentences[df_sentences['len'] > 5]

            filtered_df = df_sentences[df_sentences['comment'].str.contains('|'.join(identity_terms))]
            control_df = df_sentences[~df_sentences['comment'].str.contains('|'.join(identity_terms))]

            new_df_10 = pd.DataFrame()
            new_df_20 = pd.DataFrame()
            new_df_30 = pd.DataFrame()
            new_df_40 = pd.DataFrame()

            for term in identity_terms:
                temp = filtered_df[filtered_df['comment'].str.contains(term)].iloc[:10]
                new_df_10 = new_df_10.append(temp)

                temp = filtered_df[filtered_df['comment'].str.contains(term)].iloc[:20]
                new_df_20 = new_df_20.append(temp)

                temp = filtered_df[filtered_df['comment'].str.contains(term)].iloc[:30]
                new_df_30 = new_df_30.append(temp)

                temp = filtered_df[filtered_df['comment'].str.contains(term)].iloc[:40]
                new_df_40 = new_df_40.append(temp)

            print("setting is_toxic to false")
            new_df_10['is_toxic'] = False
            print ("length of extracted neutral comments: ", len(new_df_10))
            filename = os.path.join(SAVED_MODEL_DIR, "debiased_10.csv")
            new_df_10.to_csv(filename, index=False)
            control_df_10 = control_df[:len(new_df_10)]
            control_df_10['is_toxic'] = False
            filename = os.path.join(SAVED_MODEL_DIR,  "control_10.csv")
            control_df_10.to_csv(filename, index=False)

            new_df_20['is_toxic'] = False
            print ("length of extracted neutral comments: ", len(new_df_20))
            filename = os.path.join(SAVED_MODEL_DIR, "debiased_20.csv")
            new_df_20.to_csv(filename, index=False)
            control_df_20 = control_df[:len(new_df_20)]
            control_df_20['is_toxic'] = False
            filename = os.path.join(SAVED_MODEL_DIR, "control_20.csv")
            control_df_20.to_csv(filename, index=False)

            new_df_30['is_toxic'] = False
            print ("length of extracted neutral comments: ", len(new_df_30))
            filename = os.path.join(SAVED_MODEL_DIR, "debiased_30.csv")
            new_df_30.to_csv(filename, index=False)
            control_df_30 = control_df[:len(new_df_30)]
            control_df_30['is_toxic'] = False
            filename = os.path.join(SAVED_MODEL_DIR, "control_30.csv")
            control_df_30.to_csv(filename, index=False)

            new_df_40['is_toxic'] = False
            print ("length of extracted neutral comments: ", len(new_df_40))
            filename = os.path.join(SAVED_MODEL_DIR, "debiased_40.csv")
            new_df_40.to_csv(filename, index=False)
            control_df_40 = control_df[:len(new_df_40)]
            control_df_40['is_toxic'] = False
            filename = os.path.join(SAVED_MODEL_DIR, "control_40.csv")
            control_df_40.to_csv(filename, index=False)

            print("Generated debiased and control datasets")

    # initialize Hierarchical attention network
    print ("initializing model")
    model = HierarchicalAttn()
    print ("Done Initializing model")

    # Train the model
    if args.is_train:
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data.load_data(model_type=args.model_type,
                                                                                  sent_aug=args.sentence_augment)
        model.train(train_x, train_y, valid_x, valid_y, test_x, batch_size=128, epochs=5,
                    embeddings_path=EMBEDDINGS_PATH, saved_model_dir=SAVED_MODEL_DIR,
                    saved_model_filename=SAVED_MODEL_FILENAME)

        model.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)
        # Compute Min/max/mean attn wts and IDF for noun/adj combo on filtered test set
        nounadj_idf = pd.read_excel("idf_noun_adj.xlsx", sheetname='filtered_english')
        # load filtered validation data
        data_valid, valid_x, valid_y = data.load_validdata()
        print("Evaluating model on validation data..")
        overall_fpr = predict_on(model, data_valid, valid_x, valid_y, nounadj_idf)
        print ("Overall validation set FPR: ", overall_fpr)
        print("Calculating per term FPR on validation set")
        compute_fpr(model, data_valid, overall_fpr)

    # Evaluating model on entire test data
    if args.is_test:
        # load saved model
        print("loading saved model: ", SAVED_MODEL_FILENAME)
        model.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)
        nounadj_idf = pd.read_excel("idf_noun_adj.xlsx", sheetname='filtered_english')
        data_test, test_x, test_y = data.load_testdata()
        print("Evaluating model on full test dataset..")
        overall_fpr = predict_on(model, data_test, test_x, test_y, nounadj_idf)
        print ("Overall Test set FPR: ", overall_fpr)
        print("Calculating per term FPR on test set")
        compute_fpr(model, data_test, overall_fpr)

    # Evaluating model on entire test data
    if args.new_test:
        # load saved model
        print("loading saved model: ", SAVED_MODEL_FILENAME)
        model.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)
        nounadj_idf = pd.read_excel("idf_noun_adj.xlsx", sheetname='filtered_english')
        data_test, test_x, test_y = data.load_testdata()
        score = model.evaluate(test_x, test_y, batch_size=128)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    # Evaluating model on entire test data
    if args.is_test2:
        # load saved model
        print("loading saved model: ", SAVED_MODEL_FILENAME)
        model.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)
        nounadj_idf = pd.read_excel("idf_noun_adj.xlsx", sheetname='filtered_english')
        data_test, test_x, test_y = data.load_testdata("data/newtest.csv")
        print("Evaluating model on full test dataset..")
        overall_fpr = predict_on(model, data_test, test_x, test_y, nounadj_idf, True)
        print ("Overall Test set FPR: ", overall_fpr)
        print("Calculating per term FPR on test set")
        compute_fpr(model, data_test, overall_fpr)

    # Evaluating model on entire validation data
    if args.is_validation:
        # load saved model
        print("loading saved model: ", SAVED_MODEL_FILENAME)
        model.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)
        # Compute Min/max/mean attn wts and IDF for noun/adj combo on filtered test set
        nounadj_idf = pd.read_excel("idf_noun_adj.xlsx", sheetname='filtered_english')
        # load filtered validation data
        data_valid, valid_x, valid_y = data.load_validdata()
        print("Evaluating model on  validation data..")
        overall_fpr = predict_on(model, data_valid, valid_x, valid_y, nounadj_idf)
        print ("Overall validation set FPR: ", overall_fpr)
        print("Calculating per term FPR on validation set")
        compute_fpr(model, data_valid, overall_fpr)

    # Evaluating model on entire validation data
    if args.compute_auc:
        # load saved model
        print("loading saved model: ", SAVED_MODEL_FILENAME)
        model.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)
        # Compute Min/max/mean attn wts and IDF for noun/adj combo on filtered test set
        nounadj_idf = pd.read_excel("idf_noun_adj.xlsx", sheetname='filtered_english')
        # load filtered validation data
        data_valid, valid_x, valid_y = data.load_validdata()
        print("Evaluating model on  validation data..")
        overall_fpr = predict_on(model, data_valid, valid_x, valid_y, nounadj_idf, just_test=True)
        data_test, test_x, test_y = data.load_testdata()
        print("Evaluating model on full test dataset..")
        overall_fpr = predict_on(model, data_test, test_x, test_y, nounadj_idf, just_test=True)

    # print attention activation maps across sentences and words per sentence
    if args.a_test:
        # load saved model
        print("loading saved model: ", SAVED_MODEL_FILENAME)
        model.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)

        print("loading saved baseline model: ", SAVED_MODEL_FILENAME)
        base_model = HierarchicalAttn()
        base_model.load_weights('baseline_0', 'model_baseline.h5')
        nnadj_list = pd.read_excel("idf_noun_adj.xlsx", sheetname='filtered_english')
        sent_parts = list()
        idx2label = {0: 'non-toxic', 1: 'toxic'}
        activation_wts = pd.DataFrame()
        base_activation_wts = pd.DataFrame()
        output_text = []

        df_fname = os.path.join(SAVED_MODEL_DIR, args.model_type + "_falsepositives.csv")
        debiased_df = pd.read_csv(df_fname, usecols=['comment', 'TrueValue']).dropna()
        baseline_df = pd.read_csv("baseline_0/baseline_falsepositives.csv", usecols=['comment', 'TrueValue']).dropna()

        df_merged = baseline_df.merge(debiased_df.drop_duplicates(), on=['comment', 'TrueValue'],
                                      how='left', indicator=True)
        combined_df = pd.DataFrame(df_merged[df_merged['_merge'] == 'left_only'], columns=['comment', 'TrueValue'])
        combined_df['comment'] = combined_df['comment'].map(lambda x: data.clean_data(x))
        combined_df['alter'] = pd.DataFrame(combined_df['comment'].map(lambda x: check(x, nnadj_list.term.values)))
        combined_df.to_csv('test_intersection.csv')
        x = list(combined_df['comment'])
        y = list(combined_df['TrueValue'])
        z2 = pd.DataFrame()
        for idx in tqdm(range(len(x))):
            x[idx] = x[idx].decode("utf-8")
            if len(x[idx]) < 1:
                continue
            wts, pred = model.get_attn_wts_with_prediction(x[idx], normalized=False)
            base_wts, base_pred = base_model.get_attn_wts_with_prediction(x[idx], normalized=False)

            if y[idx] == 0 and base_pred == 1:
                z1 = pd.DataFrame({'comment': [x[idx]], "true_is_toxic": [y[idx]],
                                   "pred_is_toxic": [pred.astype(bool)],  "weights:": [wts]})
                z2 = z2.append(z1, ignore_index=True)

                sent_parts.append('<span style="padding:2px;"><b>Baseline [actual: %10s ; pred: %10s]</b></span>' % (
                    idx2label[y[idx]], idx2label[int(base_pred.flatten())]))
                sent_parts.append('<br>')
                sent_parts.append("\n")
                for i in range(len(base_wts)):
                    for j in range(len(base_wts[i])):
                        if j == len(base_wts[i])-1:
                            sent_parts.append(
                                '<span style="background: {}; color:#000; padding:2px; font-weight=\'bold\'">{}.</span>'.
                                    format(weight2color(base_wts[i][j][1]), base_wts[i][j][0]))

                        else:
                            sent_parts.append(
                                '<span style="background: {}; color:#000; padding:2px; font-weight=\'bold\'">{}</span>'.format(
                                    weight2color(base_wts[i][j][1]), base_wts[i][j][0]))
                        sent_parts.append("\n")
                sent_parts.append('<br><br>')
                sent_parts.append("\n")
                sent_parts.append('<span style="padding:2px;"><b>Debiased-60/20 [actual: %10s ; pred: %10s]</b></span>' % (
                    idx2label[y[idx]], idx2label[int(pred.flatten())]))
                sent_parts.append('<br>')
                for i in range(len(wts)):
                    for j in range(len(wts[i])):
                        if j == len(wts[i]) - 1:
                            sent_parts.append(
                                '<span style="background: {}; color:#000; padding:2px; font-weight=\'bold\'">{}.</span>'.format(
                                    weight2color(wts[i][j][1]), wts[i][j][0]))

                        else:
                            sent_parts.append(
                                '<span style="background: {}; color:#000; padding:2px; font-weight=\'bold\'">{}</span>'.format(
                                    weight2color(wts[i][j][1]), wts[i][j][0]))
                        sent_parts.append("\n")

                sent_parts.append('<br><br><br><br>')
                output_text = ''.join(sent_parts)

        outputfilename = os.path.join(SAVED_MODEL_DIR, "output_whatchanged.html")
        f_out = open(outputfilename, "w")
        f_out.write(output_text)
        f_out.close()
        z2.to_csv('output_wts.csv')

    # print attention activation maps across sentences and words per sentence
    if args.single_test:
        # load saved model
        print("loading saved model: ", SAVED_MODEL_FILENAME)
        model.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)

        print("loading saved baseline model: ", SAVED_MODEL_FILENAME)
        base_model = HierarchicalAttn()
        base_model.load_weights('baseline_0', 'model_baseline.h5')
        nnadj_list = pd.read_excel("idf_noun_adj.xlsx", sheetname='filtered_english')
        sent_parts = list()
        idx2label = {0: 'non-toxic', 1: 'toxic'}
        activation_wts = pd.DataFrame()
        base_activation_wts = pd.DataFrame()
        output_text = []

        text = "The percentage of American adults identifying as lesbian, gay, bisexual or transgender (LGBT) \
        increased to 4.5% in 2017, up from 4.1% in 2016 and 3.5% in 2012 when Gallup began tracking the measure. \
        The latest estimate is based on over 340,000 interviews conducted as part of Gallup's daily tracking in 2017."
        label = False
        z2 = pd.DataFrame()

        wts, pred = model.get_attn_wts_with_prediction(text, normalized=False)
        base_wts, base_pred = base_model.get_attn_wts_with_prediction(text, normalized=False)

        base = {'prediction': idx2label[int(base_pred.flatten())], 'weights': base_wts}
        debiased = {'prediction': idx2label[int(pred.flatten())], 'weights': wts}

        z1 = pd.DataFrame(
            {'comment': [text], "true_is_toxic": [label], "pred_is_toxic": [pred.astype(bool)],
             "weights:": [wts]})
        z2 = z2.append(z1, ignore_index=True)

        sent_parts.append(
            '<span style="padding:2px;"><b>Baseline [actual: %10s ; pred: %10s]</b></span>' % (
                idx2label[label], idx2label[int(base_pred.flatten())]))
        sent_parts.append('<br>')
        sent_parts.append("\n")
        for i in range(len(base_wts)):
            for j in range(len(base_wts[i])):
                if j == len(base_wts[i]) - 1:
                    sent_parts.append(
                        '<span style="background: {}; color:#000; padding:2px; font-weight=\'bold\'">{}.</span>'.format(
                            weight2color(base_wts[i][j][1]), base_wts[i][j][0]))

                else:
                    sent_parts.append(
                        '<span style="background: {}; color:#000; padding:2px; font-weight=\'bold\'">{}</span>'.format(
                            weight2color(base_wts[i][j][1]), base_wts[i][j][0]))
                sent_parts.append("\n")
        sent_parts.append('<br><br>')
        sent_parts.append("\n")
        sent_parts.append(
            '<span style="padding:2px;"><b>Debiased-60/20 [actual: %10s ; pred: %10s]</b></span>' % (
                idx2label[label], idx2label[int(pred.flatten())]))
        sent_parts.append('<br>')
        for i in range(len(wts)):
            for j in range(len(wts[i])):
                if j == len(wts[i]) - 1:
                    sent_parts.append(
                        '<span style="background: {}; color:#000; padding:2px; font-weight=\'bold\'">{}.</span>'.format(
                            weight2color(wts[i][j][1]), wts[i][j][0]))

                else:
                    sent_parts.append(
                        '<span style="background: {}; color:#000; padding:2px; font-weight=\'bold\'">{}</span>'.format(
                            weight2color(wts[i][j][1]), wts[i][j][0]))
                sent_parts.append("\n")

        sent_parts.append('<br><br><br><br>')
        output_text = ''.join(sent_parts)

        print(wts, idx2label[int(pred.flatten())])
        print(base_wts, idx2label[int(base_pred.flatten())])


    if args.merge_results:
        # dirname1 = args.dirname + '/cutoff_' + str(args.cutoff)
        # id_fname = dirname1 + '/identity_terms_' + str(args.cutoff) + ".xlsx"
        # id_terms = pd.read_excel(id_fname, sheetname='filtered', usecols=['term'])
        # wts_baseline = pd.read_excel("baseline_0/attn_wts_baseline_0.xlsx",
        #                              sheetname='fp', usecols=['term', 'mean_attnWt'])
        # wts_db_10 = pd.read_excel(dirname1 + "/debiased_10/attn_wts_debiased_10.xlsx",
        #                           sheetname='fp', usecols=['term', 'mean_attnWt'])
        # wts_db_20 = pd.read_excel(dirname1 + "/debiased_20/attn_wts_debiased_20.xlsx",
        #                           sheetname='fp', usecols=['term', 'mean_attnWt'])
        # wts_db_30 = pd.read_excel(dirname1 + "/debiased_30/attn_wts_debiased_30.xlsx",
        #                           sheetname='fp', usecols=['term', 'mean_attnWt'])
        # wts_db_40 = pd.read_excel(dirname1 + "/debiased_40/attn_wts_debiased_40.xlsx",
        #                           sheetname='fp', usecols=['term', 'mean_attnWt'])
        #
        # s1 = pd.merge(id_terms, wts_baseline, how='inner', on=['term'])
        # s2 = pd.merge(s1, wts_db_10, how='outer', on=['term']).fillna(0)
        # s2.columns = ['term', 'mean_attnWt', 'debias_10']
        # s3 = pd.merge(s2, wts_db_20, how='outer', on=['term']).fillna(0)
        # s3.columns = ['term', 'mean_attnWt', 'debias_10', 'debias_20']
        # s4 = pd.merge(s3, wts_db_30, how='outer', on=['term']).fillna(0)
        # s4.columns = ['term', 'mean_attnWt', 'debias_10', 'debias_20', 'debias_30']
        # s5 = pd.merge(s4, wts_db_40, how='outer', on=['term']).fillna(0)
        # s5.columns = ['term', 'mean_attnWt', 'debias_10', 'debias_20', 'debias_30', 'debias_40']
        # s5['diff_10'] = s5['mean_attnWt'] - s5['debias_10']
        # s5['diff_20'] = s5['mean_attnWt'] - s5['debias_20']
        # s5['diff_30'] = s5['mean_attnWt'] - s5['debias_30']
        # s5['diff_40'] = s5['mean_attnWt'] - s5['debias_40']
        #
        # fname = "attnwts_merged_" + str(args.cutoff) + ".xlsx"
        # fwriter2 = pd.ExcelWriter(os.path.join(dirname1, fname))
        # s5.to_excel(fwriter2, index=False)
        # fwriter2.save()

        id_terms = pd.read_excel('identity_terms_0.6.xlsx', sheetname='filtered', usecols=['term'])
        fpr_baseline = pd.read_csv("baseline_0/baseline_0_fpr.csv")
        fpr_db_10 = pd.read_csv('debiased_10/debiased_10_fpr.csv')
        fpr_db_20 = pd.read_csv('debiased_20/debiased_20_fpr.csv')
        fpr_db_30 = pd.read_csv('debiased_30/debiased_30_fpr.csv')
        fpr_db_40 = pd.read_csv('debiased_40/debiased_40_fpr.csv')

        # fpr_db_10 = pd.read_csv(dirname1 + "/debiased_10/debiased_10_fpr.csv")
        # fpr_db_20 = pd.read_csv(dirname1 + "/debiased_20/debiased_20_fpr.csv")
        # fpr_db_30 = pd.read_csv(dirname1 + "/debiased_30/debiased_30_fpr.csv")
        # fpr_db_40 = pd.read_csv(dirname1 + "/debiased_40/debiased_40_fpr.csv")

        fpr_baseline.columns = ['term', 'baseline_fpr', 'baseline_bias_per_term']
        fpr_db_10.columns = ['term', 'debias10_fpr', 'debias10_bias_per_term']
        fpr_db_20.columns = ['term', 'debias20_fpr', 'debias20_bias_per_term']
        fpr_db_30.columns = ['term', 'debias30_fpr', 'debias30_bias_per_term']
        fpr_db_40.columns = ['term', 'debias40_fpr', 'debias40_bias_per_term']

        # fpr_ctrl_10 = pd.read_csv(dirname + "/control_10/control_10_fpr.csv")
        # fpr_ctrl_20 = pd.read_csv(dirname + "/control_20/control_20_fpr.csv")
        if args.cutoff == 0.6:
            fpr_ctrl_20 = pd.read_csv(""
                                      "control_20/control_20_fpr.csv")
        # fpr_ctrl_40 = pd.read_csv(dirname + "/control_40/control_40_fpr.csv")
        # fpr_ctrl_10.columns = ['term', 'control10_fpr', 'control10_bias_per_term']
        # fpr_ctrl_20.columns = ['term', 'control20_fpr', 'control20_bias_per_term']
        if args.cutoff == 0.6:
            fpr_ctrl_20.columns = ['term', 'control20_fpr', 'control20_bias_per_term']
        # fpr_ctrl_40.columns = ['term', 'control40_fpr', 'control40_bias_per_term']

        s1 = pd.merge(id_terms, fpr_baseline, how='inner', on=['term'])
        s2 = pd.merge(s1, fpr_db_10, how='outer', on=['term']).fillna(0)
        s3 = pd.merge(s2, fpr_db_20, how='outer', on=['term']).fillna(0)
        s4 = pd.merge(s3, fpr_db_30, how='outer', on=['term']).fillna(0)
        s5 = pd.merge(s4, fpr_db_40, how='outer', on=['term']).fillna(0)
        if args.cutoff == 0.6:
            s51 = pd.merge(s5, fpr_ctrl_20, how='outer', on=['term']).fillna(0)

            s6 = s51[['term', 'baseline_fpr', 'debias10_fpr', 'debias20_fpr', 'debias30_fpr', 'debias40_fpr',
                      'control20_fpr', 'baseline_bias_per_term', 'debias10_bias_per_term', 'debias20_bias_per_term',
                      'debias30_bias_per_term', 'debias40_bias_per_term', 'control20_bias_per_term']]
        else:
            s6 = s5[['term', 'baseline_fpr', 'debias10_fpr', 'debias20_fpr', 'debias30_fpr', 'debias40_fpr',
                     'baseline_bias_per_term', 'debias10_bias_per_term', 'debias20_bias_per_term',
                     'debias30_bias_per_term', 'debias40_bias_per_term']]

        if args.cutoff == 0.6:
            mean_ctrl20 = s6.control20_bias_per_term.mean()
            s6.loc["Sum", "control20_fpr"] = "SUM"
        else:
            s6.loc["Sum", "debias40_fpr"] = "SUM"

        s6.loc["Sum", "baseline_bias_per_term"] = s6.baseline_bias_per_term.sum()
        s6.loc["Sum", "debias10_bias_per_term"] = s6.debias10_bias_per_term.sum()
        s6.loc["Sum", "debias20_bias_per_term"] = s6.debias20_bias_per_term.sum()
        s6.loc["Sum", "debias30_bias_per_term"] = s6.debias30_bias_per_term.sum()
        s6.loc["Sum", "debias40_bias_per_term"] = s6.debias40_bias_per_term.sum()
        if args.cutoff == 0.6:
            s6.loc["Sum", "control20_bias_per_term"] = s6.control20_bias_per_term.sum()

        mean_base = s6.baseline_bias_per_term.mean()
        mean_10 = s6.debias10_bias_per_term.mean()
        mean_20 = s6.debias20_bias_per_term.mean()
        mean_30 = s6.debias30_bias_per_term.mean()
        mean_40 = s6.debias40_bias_per_term.mean()
        if args.cutoff == 0.6:
            mean_ctrl20 = s6.control20_bias_per_term.mean()
            s6.loc["Average", "control20_fpr"] = "AVERAGE"
        else:
            s6.loc["Average", "debias40_fpr"] = "AVERAGE"
        s6.loc["Average", "baseline_bias_per_term"] = mean_base
        s6.loc["Average", "debias10_bias_per_term"] = mean_10
        s6.loc["Average", "debias20_bias_per_term"] = mean_20
        s6.loc["Average", "debias30_bias_per_term"] = mean_30
        s6.loc["Average", "debias40_bias_per_term"] = mean_40
        if args.cutoff == 0.6:
            s6.loc["Average", "control20_bias_per_term"] = mean_ctrl20
            mean_ctrl30 = s6.control20_bias_per_term.mean()
            s6.loc["Change", "control20_fpr"] = "%CHANGE"
        else:
            s6.loc["Change", "debias40_fpr"] = "%CHANGE"

        s6.loc["Change", "debias10_bias_per_term"] = (mean_10 - mean_base) / mean_base
        s6.loc["Change", "debias20_bias_per_term"] = (mean_20 - mean_base) / mean_base
        s6.loc["Change", "debias30_bias_per_term"] = (mean_30 - mean_base) / mean_base
        s6.loc["Change", "debias40_bias_per_term"] = (mean_40 - mean_base) / mean_base
        if args.cutoff == 0.6:
            s6.loc["Change", "control20_bias_per_term"] = (mean_ctrl20 - mean_base) / mean_base

        # s6 = pd.merge(s5, fpr_ctrl_10, how='outer', on=['term']).fillna(0)
        # s7 = pd.merge(s6, fpr_ctrl_20, how='outer', on=['term']).fillna(0)
        # s8 = pd.merge(s7, fpr_ctrl_30, how='outer', on=['term']).fillna(0)
        # s9 = pd.merge(s8, fpr_ctrl_40, how='outer', on=['term']).fillna(0)
        # s10 = s9[['term', 'baseline_fpr', 'debias10_fpr', 'debias20_fpr', 'debias30_fpr', 'debias40_fpr',
        # 'control10_fpr', 'control20_fpr', 'control30_fpr', 'control40_fpr', 'baseline_bias_per_term',
        # 'debias10_bias_per_term', 'debias20_bias_per_term', 'debias30_bias_per_term', 'debias40_bias_per_term',
        # 'control10_bias_per_term', 'control20_bias_per_term', 'control30_bias_per_term', 'control40_bias_per_term']]

        fname = "fpr_merged_" +" str(args.cutoff)" + ".xlsx"
        # fwriter2 = pd.ExcelWriter(os.path.join(dirname1, fname))
        fwriter2 = pd.ExcelWriter(fname)
        s6.to_excel(fwriter2, index=False)
        fwriter2.save()
        print ("Done merging results")
