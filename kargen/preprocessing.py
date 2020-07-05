from collections import Counter

import joblib
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin


def pad_nested_sequences(sequences, dtype='int32'):
    """Pads nested sequences to the same length.

    This function transforms a list of list sequences
    into a 3D Numpy array of shape `(num_samples, max_sent_len, max_word_len)`.

    Args:
        sequences: List of lists of lists.
        dtype: Type of the output sequences.

    # Returns
        x: Numpy array.
    """
    max_sent_len = 0
    max_word_len = 0
    for sent in sequences:
        max_sent_len = max(len(sent), max_sent_len)
        for word in sent:
            max_word_len = max(len(word), max_word_len)

    x = np.zeros((len(sequences), max_sent_len, max_word_len)).astype(dtype)
    for i, sent in enumerate(sequences):
        for j, word in enumerate(sent):
            x[i, j, :len(word)] = word

    return x


def load_data_and_labels(filename, encoding='utf-8'):
    sents, ner_labels, term_labels, rel_labels = [], [], [], []
    words, ners, terms, rels = [], [], [], []
    with open(filename, encoding=encoding) as f:
        for line in f:
            line = line.rstrip()
            if line:
                _, word, ner, term, rel, tail = line.split('\t')
                words.append(word)
                ners.append(ner)
                terms.append(term)
                rels.append(int(rel))
            else:
                sents.append(words)
                ner_labels.append(ners)
                term_labels.append(terms)
                rel_labels.append(rels)
                words, ners, terms, rels = [], [], [], []

    return sents, ner_labels, term_labels, rel_labels


class Vocabulary(object):
    """A vocabulary that maps tokens to ints (storing a vocabulary).

    Attributes:
        _token_count: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocabulary.
        _token2id: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        _id2token: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, max_size=None, lower=True, unk_token=True, specials=('<pad>',)):
        """Create a Vocabulary object.

        Args:
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            lower: boolean. Whether to convert the texts to lowercase.
            unk_token: boolean. Whether to add unknown token.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary. Default: ('<pad>',)
        """
        self._max_size = max_size
        self._lower = lower
        self._unk = unk_token
        self._token2id = {token: i for i, token in enumerate(specials)}
        self._id2token = list(specials)
        self._token_count = Counter()

    def __len__(self):
        return len(self._token2id)

    def add_token(self, token):
        """Add token to vocabulary.

        Args:
            token (str): token to add.
        """
        token = self.process_token(token)
        self._token_count.update([token])

    def add_documents(self, docs):
        """Update dictionary from a collection of documents. Each document is a list
        of tokens.

        Args:
            docs (list): documents to add.
        """
        for sent in docs:
            sent = map(self.process_token, sent)
            self._token_count.update(sent)

    def doc2id(self, doc):
        """Get the list of token_id given doc.

        Args:
            doc (list): document.

        Returns:
            list: int id of doc.
        """
        doc = map(self.process_token, doc)
        return [self.token_to_id(token) for token in doc]

    def id2doc(self, ids):
        """Get the token list.

        Args:
            ids (list): token ids.

        Returns:
            list: token list.
        """
        return [self.id_to_token(idx) for idx in ids]

    def build(self):
        """
        Build vocabulary.
        """
        token_freq = self._token_count.most_common(self._max_size)
        idx = len(self.vocab)
        for token, _ in token_freq:
            self._token2id[token] = idx
            self._id2token.append(token)
            idx += 1
        if self._unk:
            unk = '<unk>'
            self._token2id[unk] = idx
            self._id2token.append(unk)

    def process_token(self, token):
        """Process token before following methods:
        * add_token
        * add_documents
        * doc2id
        * token_to_id

        Args:
            token (str): token to process.

        Returns:
            str: processed token string.
        """
        if self._lower:
            token = token.lower()

        return token

    def token_to_id(self, token):
        """Get the token_id of given token.

        Args:
            token (str): token from vocabulary.

        Returns:
            int: int id of token.
        """
        token = self.process_token(token)
        return self._token2id.get(token, len(self._token2id) - 1)

    def id_to_token(self, idx):
        """token-id to token (string).

        Args:
            idx (int): token id.

        Returns:
            str: string of given token id.
        """
        return self._id2token[idx]

    @property
    def vocab(self):
        """Return the vocabulary.

        Returns:
            dict: get the dict object of the vocabulary.
        """
        return self._token2id

    @property
    def reverse_vocab(self):
        """Return the vocabulary as a reversed dict object.

        Returns:
            dict: reversed vocabulary object.
        """
        return self._id2token


class ELMoTransformer(BaseEstimator, TransformerMixin):
    """Convert a collection of raw documents to a document id matrix.

    Attributes:
        _use_char: boolean. Whether to use char feature.
        _num_norm: boolean. Whether to normalize text.
        _word_vocab: dict. A mapping of words to feature indices.
        _char_vocab: dict. A mapping of chars to feature indices.
        _label_vocab: dict. A mapping of labels to feature indices.
    """
    def __init__(self, options_file, weight_file,
                 lower=True, num_norm=True,
                 use_char=True, initial_vocab=None):
        self._num_norm = num_norm
        self._use_char = use_char
        self._word_vocab = Vocabulary(lower=lower)
        self._char_vocab = Vocabulary(lower=False)
        self._label_vocab_ner = Vocabulary(lower=False, unk_token=False)
        self._label_vocab_term = Vocabulary(lower=False, unk_token=False)

        if initial_vocab:
            self._word_vocab.add_documents([initial_vocab])
            self._char_vocab.add_documents(initial_vocab)
        self._elmo = Elmo(options_file, weight_file, 2, dropout=0)

    def get_vocab(self):
        return self._word_vocab.vocab

    def fit(self, x, y_ner, y_term):
        self._word_vocab.add_documents(x)
        self._label_vocab_ner.add_documents(y_ner)
        self._label_vocab_term.add_documents(y_term)
        if self._use_char:
            for doc in x:
                self._char_vocab.add_documents(doc)

        self._word_vocab.build()
        self._char_vocab.build()
        self._label_vocab_ner.build()
        self._label_vocab_term.build()

        return self

    def transform(self, X, y_ner=None, y_term=None, y_rel=None):
        word_ids = [self._word_vocab.doc2id(doc) for doc in X]
        word_ids = pad_sequences(word_ids, padding='post')
        char_ids = [[self._char_vocab.doc2id(w) for w in doc] for doc in X]
        char_ids = pad_nested_sequences(char_ids)
        character_ids = batch_to_ids(X)
        elmo_embeddings = self._elmo(character_ids)['elmo_representations'][1]
        elmo_embeddings = elmo_embeddings.detach().numpy()
        features = [word_ids, char_ids, elmo_embeddings]
        if not y_ner or not y_term or not y_rel: return features
        # NER
        y_ner = [self._label_vocab_ner.doc2id(doc) for doc in y_ner]
        y_ner = pad_sequences(y_ner, padding='post')
        y_ner = to_categorical(y_ner, self.label_size_ner).astype(int)
        y_ner = y_ner if len(y_ner.shape) == 3 else np.expand_dims(y_ner, axis=0)
        # TERM
        y_term = [self._label_vocab_term.doc2id(doc) for doc in y_term]
        y_term = pad_sequences(y_term, padding='post')
        y_term = to_categorical(y_term, self.label_size_term).astype(int)
        y_term = y_term if len(y_term.shape) == 3 else np.expand_dims(y_term, axis=0)
        # REL
        y_rel = pad_sequences(y_rel, padding='post')
        y_rel = np.expand_dims(y_rel, axis=2)
        return features, [y_ner, y_term, y_rel]

    def fit_transform(self, x, y_ner=None, y_term=None, **params):
        return self.fit(x, y_ner, y_term).transform(x, y_ner, y_term)

    def inverse_transform(self, y_ner, y_term, y_rel, lengths=None):
        y_ner = np.argmax(y_ner, -1)
        y_term = np.argmax(y_term, -1)
        y_rel = y_rel[:, :, 0].tolist()
        inverse_y_ner = [self._label_vocab_ner.id2doc(ids) for ids in y_ner]
        inverse_y_term = [self._label_vocab_term.id2doc(ids) for ids in y_term]
        if lengths:
            inverse_y_ner = [iy[:l] for iy, l in zip(inverse_y_ner, lengths)]
            inverse_y_term = [iy[:l] for iy, l in zip(inverse_y_term, lengths)]
            inverse_y_rel = []
            for iy, l in zip(y_rel, lengths):
                inverse_y_rel += iy[:l]
        else:
            inverse_y_rel = [y for y in y_rel]
        return inverse_y_ner, inverse_y_term, inverse_y_rel

    @property
    def word_vocab_size(self):
        return len(self._word_vocab)

    @property
    def char_vocab_size(self):
        return len(self._char_vocab)

    @property
    def label_size_ner(self):
        return len(self._label_vocab_ner)

    @property
    def label_size_term(self):
        return len(self._label_vocab_term)

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)

        return p

