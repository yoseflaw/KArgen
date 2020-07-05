# `KArgen` - Knowledge Acquisition Generalization with Multi-task Learning

`KArgen` is the generalization implementation for my Master's Thesis:

**Automatic Knowledge Acquisition for the Special Cargo Services Domain with Unsupervised Entity and Relation Extraction**

Code structure adopted from: [anago](https://github.com/Hironsan/anago)

The generalization part provides a model that can be used for entity/relation extraction from special cargo text.
The training set was created automatically via [KArgo](https://github.com/yoseflaw/KArgo). The model architecture can be seen here: 

<img src="https://github.com/yoseflaw/KArgen/blob/master/images/simplified_hmtl.png" alt="Simplified HMTL Architecture" width="400"/>

This repository contains the following folders:

* data/kargo: all datasets for NER/EE/RE in CONLL format. Multi-task modeling as proposed by [Bekoulis et al. (2018)](https://github.com/bekou/multihead_joint_entity_relation_extraction).
    * train: training sets as produced by [KArgo](https://github.com/yoseflaw/KArgo)
        * `not_terms_only`: dataset contains all sentences, including sentences without entities (for EE)
        * `terms_only`: dataset contains only sentences with at least one entity (for EE)
    * dev_rel, test_rel: development and test set 1
    * online_rel: test set 2 (online documents, based on HTML/PDF excerpts)
* kargen: source code folder for KArgen
    * crf.py: CRF layer implementation for Keras, based on [keras-contrib](https://github.com/keras-team/keras-contrib)
    * models.py: model structure and wrapper for simplified Hiearchical Multi-task Learning from [hmtl](https://github.com/huggingface/hmtl)
    * preprocessing.py: preprocessing pipeline for sequential deep learning model
    * trainer.py: training routine for KArgen model, including callbacks.
* main.py: example of KArgen training and evaluation routine, including saving/loading models.
* infer.ipynb: example of extraction with the trained models, visualization with [displaCy](https://explosion.ai/demos/displacy)
* results.ipynb: notebook for visualizing model training/evaluation results, can be seen here

<img src="https://github.com/yoseflaw/KArgen/blob/master/images/result_viz.png" alt="Result Training Visualization" />
    
A comparison of **P**recision/**R**ecall/**F**-score for model trained with automatic training set (`Auto`) and development set (`Manual`), for test set 1 (holdout news articles):

<img src="https://github.com/yoseflaw/KArgen/blob/master/images/result_ts1.png" alt="Result Test Set 1 News" />

and for test set 2 (online documents):

<img src="https://github.com/yoseflaw/KArgen/blob/master/images/result_ts2.png" alt="Result Test Set 1 News" />