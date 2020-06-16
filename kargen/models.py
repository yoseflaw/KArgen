import json

import numpy as np
from keras import Input, Model
from keras import backend as K
from keras.models import load_model
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.layers import Embedding, TimeDistributed, Bidirectional, LSTM, Concatenate, Dropout, Dense, Conv1D, \
    GlobalMaxPooling1D
from seqeval.metrics import sequence_labeling, f1_score

from kargen.crf import CRF, crf_loss
from kargen.preprocessing import ELMoTransformer
from kargen.trainer import Trainer
from kargen.utils import load_glove, filter_embeddings


def weighted_binary_crossentropy(weight=1.):

    def _custom_loss(y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        logloss = -(y_true * K.log(y_pred) * weight + (1 - y_true) * K.log(1 - y_pred))
        return K.mean(logloss, axis=-1)

    return _custom_loss


class SequenceModel(object):

    def __init__(self,
                 word_embedding_dim=100,
                 char_embedding_dim=16,
                 word_lstm_size=64,
                 char_lstm_size=None,
                 char_cnn_num_filters=64,
                 char_cnn_filters_size=3,
                 fc_rel_dim=64,
                 emb_dropout=0.5,
                 lstm_dropout=0.2,
                 initial_vocab=None,
                 lr=0.001,
                 rel_pos_bal=9.):
        self.model = None
        self.p = None
        self.tagger = None
        self.embeddings = None
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.word_lstm_size = word_lstm_size
        self.char_lstm_size = char_lstm_size
        self.char_cnn_num_filters = char_cnn_num_filters
        self.char_cnn_filters_size = char_cnn_filters_size
        self.fc_rel_dim = fc_rel_dim
        self.emb_dropout = emb_dropout
        self.lstm_dropout = lstm_dropout
        self.initial_vocab = initial_vocab
        self.lr = lr
        self.rel_pos_bal = rel_pos_bal

    def fit(self, x_train, y_ner_train, y_term_train, y_rel_train,
            x_valid, y_ner_valid, y_term_valid, y_rel_valid,
            embeddings_file, elmo_options_file, elmo_weights_file,
            epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True):
        print("elmo")
        p = ELMoTransformer(elmo_options_file, elmo_weights_file)
        p.fit(x_train, y_ner_train, y_term_train)
        print("glove")
        embeddings = load_glove(embeddings_file)
        embeddings = filter_embeddings(embeddings, p.get_vocab(), self.word_embedding_dim)
        print("building model")
        model = MultiLayerLSTM(
            num_labels_ner=p.label_size_ner,
            num_labels_term=p.label_size_term,
            word_vocab_size=p.word_vocab_size,
            char_vocab_size=p.char_vocab_size,
            word_embedding_dim=self.word_embedding_dim,
            char_embedding_dim=self.char_embedding_dim,
            word_lstm_size=self.word_lstm_size,
            char_lstm_size=self.char_lstm_size,
            char_cnn_num_filters=self.char_cnn_num_filters,
            char_cnn_filters_size=self.char_cnn_filters_size,
            fc_rel_dim=self.fc_rel_dim,
            emb_dropout=self.emb_dropout,
            lstm_dropout=self.lstm_dropout,
            rel_pos_bal=self.rel_pos_bal,
            embeddings=embeddings
        )
        model, loss = model.build()
        model.compile(loss=loss, optimizer=Adam(lr=self.lr))
        print("training")
        trainer = Trainer(model, preprocessor=p)
        trainer.train(
            x_train, y_ner_train, y_term_train, y_rel_train,
            x_valid, y_ner_valid, y_term_valid, y_rel_valid,
            epochs=epochs, batch_size=batch_size,
            verbose=verbose, callbacks=callbacks,
            shuffle=shuffle
        )
        self.p = p
        self.model = model

    def predict(self, x_test):
        """Returns the prediction of the model on the given test data.

        Args:
            x_test : array-like, shape = (n_samples, sent_length)
            Test samples.

        Returns:
            y_pred : array-like, shape = (n_samples, sent_length)
            Prediction labels for x.
        """
        if self.model:
            lengths = map(len, x_test)
            x_test = self.p.transform(x_test)
            y_pred = self.model.predict(x_test)
            y_pred = self.p.inverse_transform(y_pred, lengths)
            return y_pred
        else:
            raise OSError('Could not find a model. Call load(dir_path).')

    def score(self, x_test, y_test):
        """Returns the f1-micro score on the given test data and labels.

        Args:
            x_test : array-like, shape = (n_samples, sent_length)
            Test samples.

            y_test : array-like, shape = (n_samples, sent_length)
            True labels for x.

        Returns:
            score : float, f1-micro score.
        """
        if self.model:
            x_test = self.p.transform(x_test)
            lengths = map(len, y_test)
            y_pred = self.model.predict(x_test)
            y_pred = self.p.inverse_transform(y_pred, lengths)
            score = f1_score(y_test, y_pred)
            return score
        else:
            raise OSError('Could not find a model. Call load(dir_path).')

    def analyze(self, text, tokenizer=str.split):
        """Analyze text and return pretty format.

        Args:
            text: string, the input text.
            tokenizer: Tokenize input sentence. Default tokenizer is `str.split`.

        Returns:
            res: dict.
        """
        if not self.tagger:
            self.tagger = Tagger(self.model,
                                 preprocessor=self.p,
                                 tokenizer=tokenizer)
        return self.tagger.analyze(text)

    def save(self, weights_file, preprocessor_file, params_file=None):
        self.model.save(weights_file)
        self.p.save(preprocessor_file)
        if params_file:
            with open(params_file, 'w') as f:
                params = self.model.to_json()
                json.dump(json.loads(params), f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, weights_file, preprocessor_file):
        self = cls()
        self.p = ELMoTransformer.load(preprocessor_file)
        self.model = load_model(weights_file, custom_objects={'CRF': CRF}, compile=False)
        self.model.compile(loss=[crf_loss, crf_loss, weighted_binary_crossentropy(9.)], optimizer="adam")
        return self


class MultiLayerLSTM(object):
    """
    A Keras implementation of ELMo BiLSTM-CRF for sequence labeling.
    """

    def __init__(self,
                 num_labels_ner,
                 num_labels_term,
                 word_vocab_size,
                 char_vocab_size,
                 word_embedding_dim,
                 char_embedding_dim,
                 word_lstm_size,
                 char_lstm_size,
                 char_cnn_num_filters,
                 char_cnn_filters_size,
                 fc_rel_dim,
                 emb_dropout,
                 lstm_dropout,
                 rel_pos_bal,
                 embeddings):
        self._char_embedding_dim = char_embedding_dim
        self._word_embedding_dim = word_embedding_dim
        self._char_lstm_size = char_lstm_size
        self._char_cnn_num_filters = char_cnn_num_filters
        self._char_cnn_filters_size = char_cnn_filters_size
        self._word_lstm_size = word_lstm_size
        self._char_vocab_size = char_vocab_size
        self._word_vocab_size = word_vocab_size
        self._fc_rel_dim = fc_rel_dim
        self._emb_dropout = emb_dropout
        self._lstm_dropout = lstm_dropout
        self._rel_pos_bal = rel_pos_bal
        self._embeddings = embeddings
        self._num_labels_ner = num_labels_ner
        self._num_labels_term = num_labels_term

    def build(self):
        # build word embedding
        word_ids = Input(batch_shape=(None, None), dtype='int32', name='word_input')
        if self._embeddings is None:
            word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                        output_dim=self._word_embedding_dim,
                                        mask_zero=True,
                                        name='word_embedding')(word_ids)
        else:
            word_embeddings = Embedding(input_dim=self._embeddings.shape[0],
                                        output_dim=self._embeddings.shape[1],
                                        mask_zero=True,
                                        weights=[self._embeddings],
                                        name='word_embedding')(word_ids)

        # build character based word embedding
        char_ids = Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
        char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                    output_dim=self._char_embedding_dim,
                                    mask_zero=self._char_lstm_size is not None,
                                    name='char_embedding')(char_ids)
        if self._char_lstm_size:
            char_embeddings = TimeDistributed(Bidirectional(LSTM(self._char_lstm_size)))(char_embeddings)
        else:
            char_embeddings = TimeDistributed(
                Conv1D(self._char_cnn_num_filters, self._char_cnn_filters_size, padding="same", activation="relu"),
                name="char_cnn"
            )(char_embeddings)
            char_embeddings = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")(char_embeddings)

        elmo_embeddings = Input(shape=(None, 1024), dtype='float32')

        g_ner = Concatenate()([word_embeddings, char_embeddings, elmo_embeddings])

        g_ner = Dropout(self._emb_dropout)(g_ner)
        lstm_ner1 = Bidirectional(
            LSTM(units=self._word_lstm_size, return_sequences=True, dropout=self._lstm_dropout), name="lstm_ner1"
        )(g_ner)
        crf_ner = CRF(self._num_labels_ner, sparse_target=False, name="crf_ner")
        pred_ner = crf_ner(lstm_ner1)

        g_term = Concatenate()([g_ner, lstm_ner1])
        lstm_term1 = Bidirectional(
            LSTM(units=self._word_lstm_size, return_sequences=True, dropout=self._lstm_dropout), name="lstm_term1"
        )(g_term)
        crf_term = CRF(self._num_labels_term, sparse_target=False, name="crf_term")
        pred_term = crf_term(lstm_term1)

        g_rel = Concatenate()([g_term, lstm_term1])
        lstm_rel1 = Bidirectional(
            LSTM(units=self._word_lstm_size, return_sequences=True, dropout=self._lstm_dropout), name="lstm_rel1"
        )(g_rel)
        fc_rel = Dense(self._fc_rel_dim, activation="relu")(lstm_rel1)
        cls_rel = TimeDistributed(Dense(1, activation="sigmoid"), name="cls_rel")
        pred_rel = cls_rel(fc_rel)

        preds = [pred_ner, pred_term, pred_rel]
        losses = [crf_ner.loss_function, crf_term.loss_function, weighted_binary_crossentropy(self._rel_pos_bal)]
        model = Model(inputs=[word_ids, char_ids, elmo_embeddings], outputs=preds)

        print(model.summary())

        return model, losses


class Tagger(object):
    """A model API that tags input sentence.

    Attributes:
        model: Model.
        preprocessor: Transformer. Preprocessing data for feature extraction.
        tokenizer: Tokenize input sentence. Default tokenizer is `str.split`.
    """

    def __init__(self, model, preprocessor, tokenizer=str.split):
        self.model = model
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

    def predict_proba(self, text):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Args:
            text : string, the input text.

        Returns:
            y : array-like, shape = [num_words, num_classes]
            Returns the probability of the word for each class in the model,
        """
        assert isinstance(text, str)

        words = self.tokenizer(text)
        X = self.preprocessor.transform([words])
        y_ner, y_term, y_rel = self.model.predict(X)
        y_ner, y_term, y_rel = y_ner[0], y_term[0], y_rel[0]  # reduce batch dimension.
        return y_ner, y_term, y_rel

    def _get_prob(self, pred_ner, pred_term, pred_rel):
        prob_ner = np.max(pred_ner, -1)
        prob_term = np.max(pred_term, -1)
        prob_rel = np.max(pred_rel, -1)
        return prob_ner, prob_term, prob_rel

    def _get_tags(self, pred_ner, pred_term, pred_rel):
        tags_ner, tags_term, tags_rel = self.preprocessor.inverse_transform(
            np.expand_dims(pred_ner, axis=0),
            np.expand_dims(pred_term, axis=0),
            np.expand_dims(pred_rel, axis=0)
        )
        tags_ner, tags_term, tags_rel = tags_ner[0], tags_term[0], tags_rel[0]  # reduce batch dimension
        tags_rel = [int(tag > 0.5) for tag in tags_rel]
        return tags_ner, tags_term, tags_rel

    def _build_response(self, sent, tags, probs):
        words = self.tokenizer(sent)
        res = {
            'words': words,
            'entities': [],
            'terms': [],
            'head_rels': []
        }
        tag_ner, tag_term, tag_rel = tags
        prob_ner, prob_term, prob_rel = probs
        chunks_ner = sequence_labeling.get_entities(tag_ner)
        chunks_term = sequence_labeling.get_entities(tag_term)

        for chunk_type, chunk_start, chunk_end in chunks_ner:
            chunk_end += 1
            entity = {
                'text': ' '.join(words[chunk_start: chunk_end]),
                'type': chunk_type,
                'score': float(np.average(prob_ner[chunk_start: chunk_end])),
                'beginOffset': chunk_start,
                'endOffset': chunk_end
            }
            res['entities'].append(entity)
        for chunk_type, chunk_start, chunk_end in chunks_term:
            chunk_end += 1
            term = {
                'text': ' '.join(words[chunk_start: chunk_end]),
                'type': chunk_type,
                'score': float(np.average(prob_ner[chunk_start: chunk_end])),
                'beginOffset': chunk_start,
                'endOffset': chunk_end
            }
            res['terms'].append(term)
        for i, tag in enumerate(tag_rel):
            if tag:
                rel = {
                    'text': words[i],
                    'score': f"{round(prob_rel[i], 4)}",
                    'offset': i
                }
                res['head_rels'].append(rel)
        return res

    def analyze(self, text):
        preds = self.predict_proba(text)
        tags = self._get_tags(*preds)
        probs = self._get_prob(*preds)
        res = self._build_response(text, tags, probs)
        return res

    def predict(self, text):
        preds = self.predict_proba(text)
        tags = self._get_tags(*preds)
        return tags
