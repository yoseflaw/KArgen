import warnings; warnings.filterwarnings(action='ignore', category=Warning)
from kargen.models import MultiLayerLSTM, SequenceModel
from kargen.preprocessing import load_data_and_labels, ELMoTransformer
from kargen.trainer import Trainer
from kargen.utils import load_glove, filter_embeddings


"""
>>> from anago.utils import load_data_and_labels

>>> x_train, y_train = load_data_and_labels('train.txt')
>>> x_test, y_test = load_data_and_labels('test.txt')
>>> x_train[0]
['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
>>> y_train[0]
['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
>>> model = anago.Sequence()
>>> model.fit(x_train, y_train, epochs=15)
>>> import anago
>>> model = anago.Sequence()
>>> model.fit(x_train, y_train, epochs=15)
Epoch 1/15
541/541 [==============================] - 166s 307ms/step - loss: 12.9774
>>> model.score(x_test, y_test)
0.802  # f1-micro score
# For more performance, you have to use pre-trained word embeddings.
# For now, anaGo's best score is 90.94 f1-micro score.
>>> text = 'President Obama is speaking at the White House.'
>>> model.analyze(text)
{
    "words": [
        "President",
        "Obama",
        "is",
        "speaking",
        "at",
        "the",
        "White",
        "House."
    ],
    "entities": [
        {
            "beginOffset": 1,
            "endOffset": 2,
            "score": 1,
            "text": "Obama",
            "type": "PER"
        },
        {
            "beginOffset": 6,
            "endOffset": 8,
            "score": 1,
            "text": "White House.",
            "type": "LOC"
        }
    ]
}
"""

if __name__ == "__main__":
    print("load data")
    # x_train, y_train = load_data_and_labels("data/conll2003/ner/train.txt")
    # x_valid, y_valid = load_data_and_labels("data/conll2003/ner/valid.txt")
    # x_test, y_test = load_data_and_labels("data/conll2003/ner/test.txt")
    x_dev, y_dev = load_data_and_labels("data/kargo/dev.txt")
    x_test, y_test = load_data_and_labels("data/kargo/test.txt")
    print(x_dev[0])
    print(y_dev[0])
    model = SequenceModel()
    model.fit(x_dev, y_dev, x_test, y_test,
              embeddings_file="pretrain_models/glove/glove.6B.100d.txt.gz",
              elmo_options_file="pretrain_models/elmo/2x4096_512_2048cnn_2xhighway_options.json",
              elmo_weights_file="pretrain_models/elmo/2x4096_512_2048cnn_2xhighway_weights.hdf5",
              epochs=20,
              batch_size=32)
    # test_score = model.score(x_test, y_test)
    # print(test_score)
    print("try saving model")
    model.save(
        weights_file="pretrain_models/lstm/test_terms/weights.h5",
        preprocessor_file="pretrain_models/lstm/test_terms/preprocessors.json",
        params_file="pretrain_models/lstm/test_terms/params.json",
    )
    print("try loading model")
    model = SequenceModel.load(
        weights_file="pretrain_models/lstm/test_terms/weights.h5",
        preprocessor_file="pretrain_models/lstm/test_terms/preprocessors.json"
    )
    print("try analyzing a sample sentence")
    # test_score = model.score(x_test, y_test)
    text = "Emirates approved CSafe RAP active temperature-controlled packaging for pharmaceuticals and life-science cargo."
    analysis = model.analyze(text)
    from pprint import pprint
    pprint(analysis)
