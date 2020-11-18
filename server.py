from __future__ import division
from __future__ import unicode_literals

import json
import logging
import os
import re
import string
import pandas as pd
import datetime
import time
from logging.handlers import RotatingFileHandler

import tensorflow as tf
from flask import Flask, request
from flask_cors import CORS
from flask_restful import reqparse, Api
from numpy.random import seed
from spacy.lang.en import English
from tensorflow import set_random_seed

from model import HierarchicalAttn

app = Flask(__name__)
api = Api(app)
CORS(app)
# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('comment')
os.chdir('/home/aruna/Documents/HierarchicalAttentionNew/')


def weight2color(brightness):
    """Converts a single (positive) attention weight to a shade of blue."""
    brightness = brightness.item()
    brightness = int(round(255 * brightness))  # convert from 0.0-1.0 to 0-255
    ints = (255, 255 - brightness, 255 - brightness)
    return 'rgba({}, {}, {}, 0.6)'.format(*ints)


def init():
    global model, graph
    seed(10987)
    set_random_seed(78901)
    df_fname1 = os.path.join('/home/aruna/Documents/HierarchicalAttentionNew/', 'debiased_20')
    model = HierarchicalAttn()
    model.load_weights(df_fname1, 'model_debiased.h5')
    graph = tf.get_default_graph()


@app.route("/predict", methods=["POST", "GET"])
def predict():
    args = parser.parse_args()
    # text = str(args['comment'].encode('utf-8'))
    text = str(args['comment'])
    print(text)

    idx2label = {0: 'Non-Toxic', 1: 'Toxic'}
    with graph.as_default():
        wts, pred = model.activation_maps(text, normalized=False)

    sent_parts_db = list()
    if pred == 0:
        sent_parts_db.append('<span style="padding:2px; font-weight: bold;color: green">Model Prediction: %s</span>' % (
            idx2label[int(pred.flatten())]))
    if pred == 1:
        sent_parts_db.append('<span style="padding:2px; font-weight: bold; color: red">Model Prediction: %s</span>' % (
            idx2label[int(pred.flatten())]))
    sent_parts_db.append('<br>')
    sent_parts_db.append("\n")
    for i in range(len(wts)):
        for j in range(len(wts[i])):
            if j == len(wts[i]) - 2:
                if wts[i][j + 1][0] in string.punctuation:
                    sent_parts_db.append('<span style="background: {}; color:#000; ">{}{} </span>'.
                                         format(weight2color(wts[i][j][1]), wts[i][j][0],  wts[i][j+1][0]))
                    break
                else:
                    sent_parts_db.append('<span style="background: {}; color:#000; ">{}</span>'.
                                         format(weight2color(wts[i][j][1]), wts[i][j][0]))
            else:
                sent_parts_db.append('<span style="background: {}; color:#000; ">{}</span>'.
                                     format(weight2color(wts[i][j][1]), wts[i][j][0]))

            sent_parts_db.append("\n")

    output_text_db = ''.join(sent_parts_db)
    print("prediction = ", pred[0])
    return output_text_db


@app.route("/chromeextn", methods=["POST", "GET"])
def chromeextn():
    ret_json = []
    logging.basicConfig(format='%(asctime)s %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
                        level=logging.DEBUG)

    rthandler = RotatingFileHandler('extension.log', maxBytes=10000000, backupCount=10)
    rthandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)-12s  %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s')
    rthandler.setFormatter(formatter)
    logging.getLogger('').addHandler(rthandler)

    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    comment = request.args.get('comment', None)
    text = str(comment.encode('utf-8'))

    # ip_addr = request.args.get('ip', None)
    # email_addr = request.args.get('user', None)
    # docname = "GoogleDocs - docname: " + request.args.get('docname', None)
    # st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    doc = nlp(unicode(text, "utf-8"))
    sentences_temp = [sent.string.strip() for sent in doc.sents]
    sentences = [sent.replace(u'\xa0', u' ') for sent in sentences_temp]

    html = list()
    index = list()
    word_count = 0
    logging.debug(sentences)
    toxic_wordlist = list()

    for sent in sentences:
        words = sent.split()
        sent = sent.encode('ascii', errors='ignore')
        hyphenated_word = re.findall(r'\w+-\w+[-\w+]*', sent.lower())

        with graph.as_default():
            wts_orig, pred = model.get_attn_wts_with_prediction(sent.encode('utf-8'), normalized=False)
            print("prediction = ", pred[0])

        wts = []
        counter = 0
        for _, newword in enumerate(words):
            if counter >= len(wts_orig[0]):
                wts.append((newword, 0.0))
                break
            newwt = wts_orig[0][counter]
            if newword.encode('ascii', 'replace').lower().strip(string.punctuation) == newwt[0]:
                wts.append((newword, newwt[1]))
                counter += 1
            else:
                if newword.encode('ascii', 'replace').lower().strip(string.punctuation) in hyphenated_word:
                    wts.append((newword, (newwt[1] + wts_orig[0][counter + 1][1]) / 2))
                    counter += 2
                else:
                    wts.append((newword, newwt[1]))
                    counter += 1

        max_weight = 0
        index_target = 0
        html_to_return = ""

        if pred[0] == 1:
            for i in range(1):
                for j in range(len(wts)):
                    if j == len(wts) - 2:
                        if wts[j + 1][0] in string.punctuation:
                            if wts[j][1] > max_weight:
                                html_to_return = '<span style="background:{}; color:#000;">{}{}</span>'.format(
                                    weight2color(wts[j][1]), wts[j][0], wts[j + 1][0])
                                index_target = j + word_count
                                max_weight = wts[j][1]
                                toxic = wts[j+1][0]
                            break
                        else:
                            if wts[j][1] > max_weight:
                                html_to_return = '<span style="background:{}; color:#000;">{}</span>'.format(
                                    weight2color(wts[j][1]), wts[j][0])
                                index_target = j + word_count
                                max_weight = wts[j][1]
                                toxic = wts[j][0]
                    else:
                        if wts[j][1] > max_weight:
                            html_to_return = '<span style="background:{}; color:#000;">{}</span>'.format(
                                weight2color(wts[j][1]), wts[j][0])
                            index_target = j + word_count
                            max_weight = wts[j][1]
                            toxic = wts[j][0]
            html.append(html_to_return)
            toxic_wordlist.append(toxic.encode('utf-8'))
            index.append(index_target)
            word_count += len(wts)
        else:
            word_count += len(wts)

    ret_json = [{"index": t, "html": s} for t, s in zip(index, html)]
    logging.debug(ret_json)

    # if len(toxic_wordlist) > 0:
    #     df = pd.DataFrame({"User/EMail ID": email_addr, "IP Address": ip_addr,
    #                        "Date/Time": st, "Mode": docname,
    #                        "Toxic Words Used": [toxic_wordlist]})
    #
    #     if not os.path.isfile('report.csv'):
    #         df.to_csv('report.csv', header='column_names', index=False)
    #     else:  # else it exists so append without writing the header
    #         df.to_csv('report.csv', mode='a', header=False, index=False)

    return json.dumps(ret_json)


@app.route("/gmailextn", methods=["POST", "GET"])
def gmailextn():
    ret_json = []
    logging.basicConfig(format='%(asctime)s %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
                        level=logging.DEBUG)

    rthandler = RotatingFileHandler('extension.log', maxBytes=10000000, backupCount=10)
    rthandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)-12s  %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s')
    rthandler.setFormatter(formatter)
    logging.getLogger('').addHandler(rthandler)

    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    #
    # comment = request.args.get('comment', None)
    # text = str(comment.encode('utf-8'))
    # ip_addr = request.args.get('ip', None)
    # email_addr = request.args.get('user', None)
    # st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    comment = request.args.get('comment', None)
    text = str(comment.encode('utf-8'))
    # ip_addr = request.args.get('ip', None)
    # email_addr = request.args.get('user', None)
    # docname = "Gmail"
    # st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    doc = nlp(unicode(text, "utf-8"))
    sentences = [sent.string.strip() for sent in doc.sents]
    html = list()
    index = list()
    word_count = 0
    logging.debug(sentences)
    toxic_wordlist = list()

    for sent in sentences:
        words = sent.split()
        sent = sent.encode('ascii', errors='ignore')
        hyphenated_word = re.findall(r'\w+-\w+[-\w+]*', sent.lower())

        with graph.as_default():
            wts_orig, pred = model.get_attn_wts_with_prediction(sent.encode('utf-8'), normalized=False)
            print("prediction = ", pred[0])

        wts = []
        counter = 0
        for _, newword in enumerate(words):
            if counter >= len(wts_orig[0]):
                wts.append((newword, 0.0))
                break
            newwt = wts_orig[0][counter]
            if newword.encode('ascii', 'replace').lower().strip(string.punctuation) == newwt[0]:
                wts.append((newword, newwt[1]))
                counter += 1
            else:
                if newword.encode('ascii', 'replace').lower().strip(string.punctuation) in hyphenated_word:
                    wts.append((newword, (newwt[1] + wts_orig[0][counter + 1][1]) / 2))
                    counter += 2
                else:
                    wts.append((newword, newwt[1]))
                    counter += 1

        max_weight = 0
        index_target = 0
        html_to_return = ""

        if pred[0] == 1:
            for i in range(1):
                for j in range(len(wts)):
                    if j == len(wts) - 2:
                        if wts[j + 1][0] in string.punctuation:
                            if wts[j][1] > max_weight:
                                html_to_return = '<span style="background:{}; color:#000;">{}{}</span>'.format(
                                    weight2color(wts[j][1]), wts[j][0], wts[j + 1][0])
                                index_target = j + word_count
                                max_weight = wts[j][1]
                                toxic = wts[j][0]
                            break
                        else:
                            if wts[j][1] > max_weight:
                                html_to_return = '<span style="background:{}; color:#000;">{}</span>'.format(
                                    weight2color(wts[j][1]), wts[j][0])
                                index_target = j + word_count
                                max_weight = wts[j][1]
                                toxic = wts[j][0]
                    else:
                        if wts[j][1] > max_weight:
                            html_to_return = '<span style="background:{}; color:#000;">{}</span>'.format(
                                weight2color(wts[j][1]), wts[j][0])
                            index_target = j + word_count
                            max_weight = wts[j][1]
                            toxic = wts[j][0]
            html.append(html_to_return)
            toxic_wordlist.append(toxic.encode('utf-8'))
            index.append(index_target)
            word_count += len(wts)
        else:
            word_count += len(wts)

    ret_json = [{"index": t, "html": s} for t, s in zip(index, html)]
    logging.debug(ret_json)

    # if len(toxic_wordlist) > 0:
    #     df = pd.DataFrame({"User/EMail ID": email_addr, "IP Address": ip_addr,
    #                        "Date/Time": st, "Mode": docname,
    #                        "Toxic Words Used": [toxic_wordlist]})
    #
    #     if not os.path.isfile('report.csv'):
    #         df.to_csv('report.csv', header='column_names', index=False)
    #     else:  # else it exists so append without writing the header
    #         df.to_csv('report.csv', mode='a', header=False, index=False)

    return json.dumps(ret_json)


if __name__ == '__main__':
    init()
    app.run(threaded=True, port=8080)
