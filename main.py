import json
import warnings; warnings.filterwarnings(action='ignore', category=Warning)
from kargen.models import SequenceModel
from kargen.preprocessing import load_data_and_labels

if __name__ == "__main__":
    print("load data")
    x_train, y_ner_train, y_term_train, y_rel_train = load_data_and_labels(f"data/kargo/train/kpm_terms_only.txt")
    x_dev, y_ner_dev, y_term_dev, y_rel_dev = load_data_and_labels("data/kargo/dev_rel.txt")
    x_test, y_ner_test, y_term_test, y_rel_test = load_data_and_labels("data/kargo/test_rel.txt")
    x_online, y_ner_online, y_term_online, y_rel_online = load_data_and_labels("data/kargo/online_rel.txt")
    print(x_train[0])
    print(y_ner_train[0])
    print(y_term_train[0])
    print(y_rel_train[0])
    model = SequenceModel(lr=5e-4)
    history = model.fit(x_train, y_ner_train, y_term_train, y_rel_train,
                        x_test, y_ner_test, y_term_test, y_rel_test,
                        embeddings_file="pretrain_models/glove/glove.6B.100d.txt.gz",
                        elmo_options_file="pretrain_models/elmo/2x4096_512_2048cnn_2xhighway_options.json",
                        elmo_weights_file="pretrain_models/elmo/2x4096_512_2048cnn_2xhighway_weights.hdf5",
                        steps_per_epoch=18,
                        epochs=20,
                        batch_size=32,
                        verbose=1)
    print(history)
    print("try saving model")
    model.save(
        weights_file=f"pretrain_models/lstm/spe18_20e/kpm/weights.h5",
        preprocessor_file=f"pretrain_models/lstm/spe18_20e/kpm/preprocessors.json",
        params_file=f"pretrain_models/lstm/spe18_20e/kpm/params.json",
    )
    print("try loading model")
    model = SequenceModel.load(
        weights_file="pretrain_models/lstm/spe18_20e/kpm/weights.h5",
        preprocessor_file="pretrain_models/lstm/spe18_20e/kpm/preprocessors.json"
    )
    print("try resuming")
    history = model.resume(x_train, y_ner_train, y_term_train, y_rel_train,
                           x_online, y_ner_online, y_term_online, y_rel_online,
                           epochs=1, steps_per_epoch=1, batch_size=32)
    with open(f"logs/results/kpm_20e.json", "w") as f:
        hist = history.history
        json.dump(hist, f, indent=2)
    print("try saving model")
    model.save(
        weights_file=f"pretrain_models/lstm/spe18_40e/kpm/weights.h5",
        preprocessor_file=f"pretrain_models/lstm/spe18_40e/kpm/preprocessors.json",
        params_file=f"pretrain_models/lstm/spe18_40e/kpm/params.json",
    )
    with open(f"logs/results/kpm_20-40e.json", "w") as f:
        hist = history.history
        json.dump(hist, f, indent=2)
    print("try analyzing a sample sentence")
    news1 = "Emirates approved CSafe RAP active temperature-controlled packaging for pharmaceuticals and life-science cargo."
    news2 = "The new facility, opened today (October 23), has 3,620 sq m of temperature-controlled warehouse space."
    news3 = "The KT400D-60 container is capable of maintaining temperatures between -60 -80 degrees Celsius for 120 hours, " \
            "with an ambient temperature of 30 degrees Celsius."
    online1 = "Temperature controlled dolly, protecting shipment from extreme heat on tarmac at Hong Kong International Airport"
    online2 = "Active container – supported by Envirotainer and CSafe"
    online3 = """
    You can count on your valuable shipments getting to their destination reliably, securely and intact, which we ensure with __customized solutions__, __precise temperature control__, the highest security standards and more.\u2022  Special  infrastructure: a state-of-the-art Cool Center, where pharmaceutical shipments, classic perishables and temperature-controlled dangerous goods are stored separately\u2022  Temperature-stable  transport: a temperature- controlled, optimized cold chain thanks to  significantly shorter transport routes and fewer interruptions during transfers, as well as  temperature-controlled compartments on board  most flights\u2022  Cool  service: a range of different cooling contain-ers that can also be made available outside of flight and processing hours\u2022 Just in case: a combination of Cool together with Care, our special product for dangerous goods, which complies with the IATA Dangerous Goods Regulations\u2022  Comprehensive  documentation: transparent pro-cesses with continuous shipment monitoring and password-protected shipment tracking in real time\u2022 Personal and competent: teams of experts ensure that pharmaceutical and medical technology ship-ments are monitored and handled appropriately
    """
    analysis = model.analyze(online3)
    print(analysis)
