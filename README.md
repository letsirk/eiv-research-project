# Research Project in Machine Learning and Data Science (07/2019)

## Introduction

A research project, continuous to [project](https://github.com/letsirk/EIV) in Explorative Information Visualization course, explores different Machine Learning and Natural Language Processing techniques to group ingredients based on the names in the receipts. The project was accomplished during 2019 for *Research Project in Machine Learning and Data Science* course. The detailed information about the project and its results can be found [here](https://github.com/letsirk/eiv-research-project/blob/master/research-project-in-machine-learning-report.pdf). Furthermore, the results are summarized in the last cell of [this notebook](https://letsirk.github.io/eiv-research-project/).

## Techniques and Tools

The research project utilizes following techniques and tools: 
* Python and Jupyter Notebook 
  * [scikit-learn](https://scikit-learn.org/stable/) library to access Machine Learning models
  * other general libaries to show (pandas, seaborn, matplotlib) or manipulate (numpy) data
  * [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) library to access a NLP model from Gensim
  * [wvlib](https://github.com/fginter/wvlib_light/tree/3471a8db66883769c4e5398806876d5be3e3df24) library to access a pretrained NLP model from [Turku NLP](https://turkunlp.org/finnish_nlp.html)

All the following models were trained, validated and tested:
* Machine Learning models
  * Decision Tree
  * Random Forest
  * K-means++
* Natural Language Processing models
  * Turku NLP pretrained model
  * Gensim Word2Vec
