import math
import numpy as np
from seqeval.metrics import f1_score as seqeval_f1, classification_report as seqeval_cr
from sklearn.metrics import f1_score as sklearn_f1, classification_report as sklearn_cr
from keras.utils import Sequence
from keras.callbacks import Callback
from pprint import pprint


class NERSequence(Sequence):

    def __init__(self, x, y_ner, y_term, y_rel, batch_size=1, preprocess=None):
        self.x = x
        self.y_ner = y_ner
        self.y_term = y_term
        self.y_rel = y_rel
        self.batch_size = batch_size
        self.preprocess = preprocess

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y_ner = self.y_ner[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y_term = self.y_term[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y_rel = self.y_rel[idx * self.batch_size: (idx + 1) * self.batch_size]
        return self.preprocess(batch_x, batch_y_ner, batch_y_term, batch_y_rel)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


class CallbackScore(Callback):

    def __init__(self, seq, preprocessor=None):
        super(CallbackScore, self).__init__()
        self.seq = seq
        self.p = preprocessor

    def get_lengths(self, y_true):
        lengths = []
        for y in np.argmax(y_true, -1):
            try:
                i = list(y).index(0)
            except ValueError:
                i = len(y)
            lengths.append(i)

        return lengths

    def on_epoch_end(self, epoch, logs=None):
        if logs is None: logs = {}
        label_true_ner, label_true_term, label_true_rel = [], [], []
        label_pred_ner, label_pred_term, label_pred_rel = [], [], []
        for i in range(len(self.seq)):
            x_true, (y_true_ner, y_true_term, y_true_rel) = self.seq[i]
            lengths = self.get_lengths(y_true_ner)
            y_true_ner, y_true_term, y_true_rel = self.p.inverse_transform(y_true_ner, y_true_term, y_true_rel, lengths)
            label_true_ner.extend(y_true_ner)
            label_true_term.extend(y_true_term)
            label_true_rel.extend(y_true_rel)
            y_pred_ner, y_pred_term, y_pred_rel = self.model.predict_on_batch(x_true)
            y_pred_ner, y_pred_term, y_pred_rel = self.p.inverse_transform(y_pred_ner, y_pred_term, y_pred_rel, lengths)
            label_pred_ner.extend(y_pred_ner)
            label_pred_term.extend(y_pred_term)
            label_pred_rel.extend(y_pred_rel)

        ner_score = seqeval_f1(label_true_ner, label_pred_ner)
        print(' - NER f1: {:04.2f}'.format(ner_score * 100))
        print(seqeval_cr(label_true_ner, label_pred_ner))
        logs['f1_ner'] = ner_score
        term_score = seqeval_f1(label_true_term, label_pred_term)
        print(' - TERM f1: {:04.2f}'.format(term_score * 100))
        print(seqeval_cr(label_true_term, label_pred_term))
        logs['f1_term'] = term_score
        label_pred_class = [pred > 0.5 for pred in label_pred_rel]
        rel_score = sklearn_f1(label_true_rel, label_pred_class)
        print(' - REL F1: {:04.2f}'.format(rel_score * 100))
        print(sklearn_cr(label_true_rel, label_pred_class))
        logs['f1_rel'] = rel_score


class Trainer(object):
    """A trainer that train the model.

    Attributes:
        _model: Model.
        _preprocessor: Transformer. Preprocessing data for feature extraction.
    """

    def __init__(self, model, preprocessor=None):
        self._model = model
        self._preprocessor = preprocessor

    def train(self, x_train, y_ner_train, y_term_train, y_rel_train,
              x_valid=None, y_ner_valid=None, y_term_valid=None, y_rel_valid=None,
              steps_per_epoch=None, epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True):
        train_seq = NERSequence(
            x_train, y_ner_train, y_term_train, y_rel_train,
            batch_size, self._preprocessor.transform
        )

        # if x_valid and y_term_valid:
        valid_seq = NERSequence(
            x_valid, y_ner_valid, y_term_valid, y_rel_valid,
            batch_size, self._preprocessor.transform
        )
        callback_score = CallbackScore(valid_seq, preprocessor=self._preprocessor)
        callbacks = [callback_score] + callbacks if callbacks else [callback_score]

        return self._model.fit_generator(generator=train_seq,
                                         validation_data=valid_seq,
                                         steps_per_epoch=steps_per_epoch,
                                         epochs=epochs,
                                         callbacks=callbacks,
                                         verbose=verbose,
                                         shuffle=shuffle)
