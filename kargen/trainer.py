import math
import numpy as np
from seqeval.metrics import f1_score, classification_report
from keras.utils import Sequence
from keras.callbacks import Callback


class NERSequence(Sequence):

    def __init__(self, x, y_ner, y_term, batch_size=1, preprocess=None):
        self.x = x
        self.y_ner = y_ner
        self.y_term = y_term
        self.batch_size = batch_size
        self.preprocess = preprocess

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y_ner = self.y_ner[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y_term = self.y_term[idx * self.batch_size: (idx + 1) * self.batch_size]

        return self.preprocess(batch_x, batch_y_ner, batch_y_term)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


class F1score(Callback):

    def __init__(self, seq, preprocessor=None):
        super(F1score, self).__init__()
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
        label_true_ner, label_true_term = [], []
        label_pred_ner, label_pred_term = [], []
        for i in range(len(self.seq)):
            x_true, (y_true_ner, y_true_term) = self.seq[i]
            lengths = self.get_lengths(y_true_ner)
            y_true_ner, y_true_term = self.p.inverse_transform(y_true_ner, y_true_term, lengths)
            label_true_ner.extend(y_true_ner)
            label_true_term.extend(y_true_term)
            y_pred_ner, y_pred_term = self.model.predict_on_batch(x_true)
            y_pred_ner, y_pred_term = self.p.inverse_transform(y_pred_ner, y_pred_term, lengths)
            label_pred_ner.extend(y_pred_ner)
            label_pred_term.extend(y_pred_term)

        ner_score = f1_score(label_true_ner, label_pred_ner)
        print(' - NER f1: {:04.2f}'.format(ner_score * 100))
        print(classification_report(label_true_ner, label_pred_ner))
        logs['f1_ner'] = ner_score
        term_score = f1_score(label_true_term, label_pred_term)
        print(' - TERM f1: {:04.2f}'.format(term_score * 100))
        print(classification_report(label_true_term, label_pred_term))
        logs['f1_term'] = term_score


class Trainer(object):
    """A trainer that train the model.

    Attributes:
        _model: Model.
        _preprocessor: Transformer. Preprocessing data for feature extraction.
    """

    def __init__(self, model, preprocessor=None):
        self._model = model
        self._preprocessor = preprocessor

    def train(self, x_train, y_ner_train, y_term_train,
              x_valid=None, y_ner_valid=None, y_term_valid=None,
              epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True):
        train_seq = NERSequence(x_train, y_ner_train, y_term_train, batch_size, self._preprocessor.transform)

        if x_valid and y_term_valid:
            valid_seq = NERSequence(x_valid, y_ner_valid, y_term_valid, batch_size, self._preprocessor.transform)
            f1 = F1score(valid_seq, preprocessor=self._preprocessor)
            callbacks = [f1] + callbacks if callbacks else [f1]

        self._model.fit_generator(generator=train_seq,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  verbose=verbose,
                                  shuffle=shuffle)
