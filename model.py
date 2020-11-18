import pickle
import os
import re
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import CustomObjectScope
from sklearn.metrics import roc_auc_score

from data import load_glove_embedding, normalize, tag, process_sent

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


class RocAucEval(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC on validation data - epoch: %d - score: %.6f \n" % (epoch + 1, score))


class Attention(Layer):
    def __init__(self, regularizer=None, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.regularizer = regularizer
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.context = self.add_weight(name='context', shape=(input_shape[-1], 1), initializer=self.init,
                                       regularizer=self.regularizer, trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        attention_in = K.exp(K.squeeze(K.dot(x, self.context), axis=-1))
        attention = attention_in / K.expand_dims(K.sum(attention_in, axis=-1), -1)

        if mask is not None:
            # use only the inputs specified by the mask
            attention = attention * K.cast(mask, 'float32')

        weighted_sum = K.batch_dot(K.permute_dimensions(x, [0, 2, 1]), attention)
        return weighted_sum

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return input_shape[0], input_shape[-1]


class HierarchicalAttn:
    def __init__(self):
        self.model = None
        self.max_sentence_len = 0
        self.max_sentence_count = 0
        self.vocab_size = 0
        self.word_embedding = None
        self.model = None
        self.word_attention_model = None
        self.tokenizer = None
        self.class_count = 2

    def build_model(self, n_classes=2, embedding_dim=300, embeddings_path=False):
        l2_reg = regularizers.l2(1e-8)
        embedding_matrix = load_glove_embedding(embeddings_path, embedding_dim, self.tokenizer.word_index)

        # Generate word-attention-weighted sentence scores
        sentence_in = Input(shape=(self.max_sentence_len,), dtype='int32')
        embedded_word_seq = Embedding(self.vocab_size, embedding_dim, weights=[embedding_matrix],
                                      input_length=self.max_sentence_len, trainable=True, name='word_embeddings', )(
            sentence_in)
        # return sequences True to return the hidden state output for each input time step.
        word_encoder = Bidirectional(CuDNNLSTM(100, return_sequences=True))(embedded_word_seq)
        dense_transform_w = Dense(200, activation='relu', name='dense_transform_w', kernel_regularizer=l2_reg)(
            word_encoder)
        attention_weighted_sentence = Model(sentence_in,
                                            Attention(name='word_attention', regularizer=l2_reg)(dense_transform_w))
        self.word_attention_model = attention_weighted_sentence
        attention_weighted_sentence.summary()

        # Generate sentence-attention-weighted document scores
        texts_in = Input(shape=(self.max_sentence_count, self.max_sentence_len), dtype='int32')
        attention_weighted_sentences = TimeDistributed(attention_weighted_sentence)(texts_in)
        sentence_encoder = Bidirectional(CuDNNLSTM(100, return_sequences=True))(attention_weighted_sentences)
        dense_transform_s = Dense(200, activation='relu', name='dense_transform_s', kernel_regularizer=l2_reg)(
            sentence_encoder)
        attention_weighted_text = Attention(name='sentence_attention', regularizer=l2_reg)(dense_transform_s)
        prediction = Dense(n_classes, activation='softmax')(attention_weighted_text)
        model = Model(texts_in, prediction)
        model.summary()

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def load_weights(self, saved_model_dir, saved_model_filename):
        with CustomObjectScope({'Attention': Attention}):
            print("#################" + os.path.join(saved_model_dir, saved_model_filename))
            self.model = load_model(os.path.join(saved_model_dir, saved_model_filename))
            self.word_attention_model = self.model.get_layer('time_distributed_1').layer
            tokenizer_path = os.path.join(saved_model_dir, saved_model_filename + '.tokenizer')
            tokenizer_state = pickle.load(open(tokenizer_path, "rb"))
            self.tokenizer = tokenizer_state['tokenizer']
            self.max_sentence_count = tokenizer_state['maxSentenceCount']
            self.max_sentence_len = tokenizer_state['maxSentenceLength']
            self.vocab_size = tokenizer_state['vocabularySize']
            self.create_reverse_word_index()

    def fit_on_texts(self, texts):
        self.tokenizer = Tokenizer(filters='"()*,-/;[\]^_`{|}~')
        all_sentences = []
        max_sentence_count = 0
        max_sentence_length = 0
        for text in texts:
            sentence_count = len(text)
            if sentence_count > max_sentence_count:
                max_sentence_count = sentence_count
            for sentence in text:
                sentence_length = len(sentence)
                if sentence_length > max_sentence_length:
                    max_sentence_length = sentence_length
                all_sentences.append(sentence)

        self.max_sentence_count = min(max_sentence_count, 20)
        self.max_sentence_len = min(max_sentence_length, 100)
        self.tokenizer.fit_on_texts(all_sentences)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.create_reverse_word_index()

    def create_reverse_word_index(self):
        self.reverse_word_index = {value: key for key, value in self.tokenizer.word_index.items()}

    def encode_texts(self, texts):
        encoded_texts = np.zeros((len(texts), self.max_sentence_count, self.max_sentence_len))
        for i, text in enumerate(texts):
            encoded_text = np.array(
                pad_sequences(self.tokenizer.texts_to_sequences(text), maxlen=self.max_sentence_len))[
                           :self.max_sentence_count]
            encoded_texts[i][-len(encoded_text):] = encoded_text
        return encoded_texts

    def save_tokenizer_on_epoch_end(self, path, epoch):
        if epoch == 0:
            tokenizer_state = {'tokenizer': self.tokenizer, 'maxSentenceCount': self.max_sentence_count,
                               'maxSentenceLength': self.max_sentence_len, 'vocabularySize': self.vocab_size}
            pickle.dump(tokenizer_state, open(path, "wb"))

    def train(self, train_x, train_y, valid_x, valid_y, test_x, batch_size, epochs, embedding_dim=300,
              embeddings_path=False, saved_model_dir='saved_models', saved_model_filename=None, ):
        # fit tokenizer
        print("fitting tokenizer")
        self.fit_on_texts(np.concatenate((train_x, valid_x, test_x)))
        encoded_train_x = self.encode_texts(train_x)
        encoded_valid_x = self.encode_texts(valid_x)

        print("building model")
        self.model = self.build_model(n_classes=train_y.shape[-1], embedding_dim=embedding_dim,
                                      embeddings_path=embeddings_path)
        roc_auc = RocAucEval(validation_data=(encoded_valid_x, valid_y), interval=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        reduce_lr = ReduceLROnPlateau()
        lambda_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: self.save_tokenizer_on_epoch_end(
            os.path.join(saved_model_dir, saved_model_filename + '.tokenizer'), epoch))

        model_checkpoint = ModelCheckpoint(filepath=os.path.join(saved_model_dir, saved_model_filename),
                                           monitor='val_acc', save_best_only=True, save_weights_only=False)
        print("training")
        self.model.fit(x=encoded_train_x, y=train_y, validation_data=(encoded_valid_x, valid_y), batch_size=batch_size,
                       epochs=epochs, verbose=1, shuffle=True,
                       callbacks=[reduce_lr, roc_auc, lambda_cb, early_stopping, model_checkpoint])

    def encode_input(self, x, spacytokenize=True):
        x = np.array(x)
        if not x.shape:
            x = np.expand_dims(x, 0)
        texts = np.array([normalize(text) for text in x])

        return self.encode_texts(texts)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y, batch_size=128):
        encoded_x = self.encode_texts(x)
        return self.model.evaluate(encoded_x, y, batch_size)

    def activation_maps(self, text, normalized=False):
        if not normalized:
            normalized_text = process_sent(text)
        else:
            normalized_text = text

        encoded_text = self.encode_input(text)[0]

        # get word activations
        hidden_word_encoding_out = Model(inputs=self.word_attention_model.input,
                                         outputs=self.word_attention_model.get_layer('dense_transform_w').output)
        hidden_word_encodings = hidden_word_encoding_out.predict(encoded_text)
        word_context = self.word_attention_model.get_layer('word_attention').get_weights()[0]
        u_wattention = encoded_text * np.exp(np.squeeze(np.dot(hidden_word_encodings, word_context)))

        # generate word, activation pairs
        nopad_encoded_text = encoded_text[-len(normalized_text):]
        nopad_encoded_text = [list(filter(lambda x: x > 0, sentence)) for sentence in nopad_encoded_text]
        reconstructed_texts = [[self.reverse_word_index[int(i)] for i in sentence] for sentence in nopad_encoded_text]
        nopad_wattention = u_wattention[-len(normalized_text):]

        # Add a very small value to avoid division by zero
        nopad_wattention = nopad_wattention / (np.expand_dims(np.sum(nopad_wattention, -1), -1) + 1e-10)
        nopad_wattention = np.array(
            [attention_seq[-len(sentence):] for attention_seq, sentence in zip(nopad_wattention, nopad_encoded_text)])
        word_activation_maps = []

        original_text = []
        filtered_sentences = process_sent(text)
        for i, sent in zip(range(len(filtered_sentences)), filtered_sentences):
            original_text.append(sent.split())
        recon_text = [[' '.join(i)] for i in reconstructed_texts]
        orig_txt = [[' '.join(i)] for i in original_text]

        matching = True
        for sent1, sent2 in zip(recon_text, orig_txt):
            if str(sent1).lower() == str(sent2).lower():
                continue
            else:
                matching = False
        if matching:
            for i, text1 in enumerate(original_text):
                for j, word in zip(range(len(text1)), text1):
                    if re.match(u"\u2019", word):
                        print(word + " has apostrophe")

                word_activation_maps.append(list(zip(text1, nopad_wattention[i])))
        else:
            for i, text1 in enumerate(reconstructed_texts):
                word_activation_maps.append(list(zip(text1, nopad_wattention[i])))

        prediction_out = np.argmax(self.predict(self.encode_input(text)), axis=1)
        return word_activation_maps, prediction_out

    def get_attn_wts_with_prediction(self, text, normalized=False):
        if not normalized:
            normalized_text = normalize(text)
        else:
            normalized_text = text

        encoded_text = self.encode_input(text.rstrip("."))[0]

        # get word activations
        hidden_word_encoding_out = Model(inputs=self.word_attention_model.input,
                                         outputs=self.word_attention_model.get_layer('dense_transform_w').output)
        hidden_word_encodings = hidden_word_encoding_out.predict(encoded_text)
        word_context = self.word_attention_model.get_layer('word_attention').get_weights()[0]
        u_wattention = encoded_text * np.exp(np.squeeze(np.dot(hidden_word_encodings, word_context)))

        # generate word, activation pairs
        nopad_encoded_text = encoded_text[-len(normalized_text):]
        nopad_encoded_text = [list(filter(lambda x: x > 0, sentence)) for sentence in nopad_encoded_text]
        reconstructed_texts = [[self.reverse_word_index[int(i)] for i in sentence] for sentence in nopad_encoded_text]
        nopad_wattention = u_wattention[-len(normalized_text):]

        # Add a very small value to avoid division by zero
        nopad_wattention = nopad_wattention / (np.expand_dims(np.sum(nopad_wattention, -1), -1) + 1e-10)
        nopad_wattention = np.array(
            [attention_seq[-len(sentence):] for attention_seq, sentence in zip(nopad_wattention, nopad_encoded_text)])
        word_activation_maps = []

        original_text = []
        filtered_sentences = process_sent(text)
        for i, sent in zip(range(len(filtered_sentences)), filtered_sentences):
            original_text.append(sent.split())
        recon_text = [[' '.join(i)] for i in reconstructed_texts]
        orig_txt = [[' '.join(i)] for i in original_text]

        matching = True
        for sent1, sent2 in zip(recon_text, orig_txt):
            if str(sent1).lower() == str(sent2).lower():
                continue
            else:
                matching = False
        if matching:
            for i, text1 in enumerate(original_text):
                for j, word in zip(range(len(text1)), text1):
                    if re.match(u"\u2019", word):
                        print(word + " has apostrophe")

                word_activation_maps.append(list(zip(text1, nopad_wattention[i])))
        else:
            for i, text1 in enumerate(reconstructed_texts):
                word_activation_maps.append(list(zip(text1, nopad_wattention[i])))

        prediction_out = np.argmax(self.predict(self.encode_input(text)), axis=1)
        return word_activation_maps, prediction_out
