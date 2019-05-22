import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

import gensim
import multiprocessing
import collections

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from datahandling import WikiDumpToCorpus

def _create_confusion_matrix(y_test, y_pred, classifier_name='Decision Tree', classes=[]):
    c_matrix = confusion_matrix(y_test.to_numpy(), y_pred, labels=classes) 
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_title("{} confusion matrix".format(classifier_name))
    sns.heatmap(c_matrix, cmap='Blues', annot=True, fmt='g', cbar=False)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('True labels')
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",rotation_mode="anchor")
    plt.savefig('{}.jpg'.format(classifier_name.replace(' ','')))

# Decision Tree
def DecisionTreeModel(X_train, y_train, X_test, y_test):
    # Create and train
    clf = DecisionTreeClassifier().fit(X_train,y_train)

    # Predict the repsonse of the train data and test data
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    return {'train score':accuracy_score(y_train, y_train_pred), # Evaluate
            'test score':accuracy_score(y_test, y_test_pred), # Evaluate
            'train pred': y_train_pred, 'train real': y_train,
            'test pred': y_test_pred, 'test real': y_test,
            'model': clf}

# Random Forest
def RandomForestModel(X_train, y_train, X_test, y_test):
    # Create and train
    clf = RandomForestClassifier(n_estimators=100).fit(X_train,y_train.squeeze())

    # Predict the repsonse of the train data and test data
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    return {'train score':accuracy_score(y_train, y_train_pred), # Evaluate
            'test score':accuracy_score(y_test, y_test_pred), # Evaluate
            'train pred': y_train_pred, 'train real': y_train,
            'test pred': y_test_pred, 'test real': y_test,
            'model': clf}

# K-means++
def _kmeans_convert_clusters_to_classes(y_real, y_pred,classes_id,classes_order=[],iter=10000):
    """ 
    Return the prediction values with best accuracy
    """
    y_pred_best =  y_pred[:]
    score_best = 0
    random_classes_best = classes_id

    for j in range(0,iter):
        # Take classes in random order if class order is not assigned
        random_classes = list(np.random.permutation(classes_id)) if len(classes_order)==0 else classes_order
        y_pred_temp = y_pred[:]

        # Assing classes to clusters with the majority elements
        for i in range(0,len(random_classes)):
            c = random_classes[i]
            target_group_indices = np.argwhere(y_real.flatten()==c).flatten()
            predicted_group = y_pred_temp[target_group_indices]
            for pred_label_temp in collections.Counter(predicted_group).most_common(): #sorted labels according to their occurences
                if pred_label_temp[0] not in random_classes[:i] and pred_label_temp[0]<50:
                    #print(pred_label_temp[0], c)
                    y_pred_temp[y_pred_temp == pred_label_temp[0]] = c
                    break
        # Assing lables that cannot be found in classes to -1
        for pred_label_temp in collections.Counter(y_pred_temp).most_common():

            if pred_label_temp[0] not in random_classes:
                y_pred_temp[y_pred_temp == pred_label_temp[0]] = -1

        # Compute accuracy score
        score = accuracy_score(y_real, y_pred)

        # Update if score was improved
        if score > score_best:
            y_pred_best = y_pred[:]
            score_best = score
            random_classes_best = random_classes 

        if classes_order: break

    return {'score': score_best, 'pred': y_pred_best, 'classes_order': random_classes_best}

def KmeansModel(X_train, y_train, X_test, y_test, classes_id):
    # Create and train
    clf = KMeans(n_clusters=len(classes_id))
    clf.fit(X_train)

    # Predict the response of the train data and test data
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    # Keep it simple and convert clusters to classes according to elements majority
    mapping_table = [-1 for i in range(0, len(classes_id))]
    y_train_most_common = collections.Counter(y_train_pred).most_common()
    y_train_pred_most_common = collections.Counter(y_train.flatten()).most_common()

    train_result = _kmeans_convert_clusters_to_classes(y_train, y_train_pred, classes_id)
    test_result = _kmeans_convert_clusters_to_classes(y_test, y_test_pred, classes_id, train_result['classes_order'])
    return {'train score': train_result['score'],
            'test score':test_result['score'], 
            'train pred': train_result['pred'], 'train real': y_train,
            'test pred':  test_result['pred'], 'test real': y_test,
            'classes_order': train_result['classes_order'],'model': clf}                                           
                                               
# NLP 
class NLPModel():
    def __init__(self, classes_id,classes_name):
        self.wv = None
        self.classes_id = classes_id[:]
        self.classes_id.append(-1) # For ingredients which were unsuccessfully identified
        self.classes_name = classes_name
        # Preprocess class names        
        self.preprocessed_classes = []
        for c in classes_name:
            temp = c.lower().replace(',','').replace('ja','')
            temp = ' '.join(temp.split()).split()
            self.preprocessed_classes.append(temp)
            
    def load(self, model_name = 'kyubyong', df_data=None):
        '''
        Available pre-trained models: lwvlib, kyubyong, wikicorpus
        '''
        if model_name == 'lwvlib':
            # http://dl.turkunlp.org/finnish-embeddings/
            # https://turkunlp.org/finnish_nlp.html --> Finnish internet parsebank
            # http://bionlp-www.utu.fi/wv_demo/ --> online interface
            # https://github.com/fginter/wvlib_light/tree/3471a8db66883769c4e5398806876d5be3e3df24 --> library
            import lwvlib
            self.wv=lwvlib.load("finnish_4B_parsebank_skgram.bin") # 
            #self.wv=lwvlib.load("finnish_s24_skgram.bin") #,10000,500000)
        elif model_name == 'kyubyong':
            #import gensim 
            #https://github.com/Kyubyong/wordvectors
            self.wv = gensim.models.Word2Vec.load('kyubyong_fin_word2vec.bin').wv
        elif model_name == 'wikicorpus':
            wiki_corpus = WikiDumpToCorpus('fiwiki-latest-pages-articles.xml.bz2')
            if os.path.exists('wikicorpus_word2vec.bin'): # load model
                self.mv = gensim.models.Word2Vec.load('wikicorpus_word2vec.bin').wv
            elif wiki_corpus.check_corpus(): # train model
                # model = gensim.models.Word2Vec(['testi'], size=200, window=10, min_count=5, 
                #             workers=multiprocessing.cpu_count(), iter=2000)
                if not df_data.empty:
                    # sentences = []
                    # for i, row in df_data.iterrows():
                    #     sentence = gensim.utils.simple_preprocess(row['Nimi'])
                    #     if len(sentence) > 0: 
                    #         sentences.append(sentence)
                    # model.build_vocab(sentences)

                    def train_func(train_data):
                        model = gensim.models.Word2Vec(gensim.utils.simple_preprocess(train_data),
                                    size=200, window=10, min_count=5, 
                                    workers=multiprocessing.cpu_count(), iter=1)
                        model.save("wikicorpus_word2vec.model")
                        model.wv.save_word2vec_format("wikicorpus_word2vec.bin", binary=True)
                        #data = gensim.utils.simple_preprocess(train_data)
                        #model.train(data, total_examples=len(data), epochs=100)
                    
                    wiki_corpus.open_corpus(train_func)
                    
                    
                
            elif os.path.exists('fiwiki-latest-pages-articles.xml.bz2'): # make corpus
                wiki_corpus.make_corpus()
            else: #train
                print("Error in 'wikicorpus'")

 
    def train(self, df_data):
        train_data = []
        for i, row in df_data.iterrows():
            # group1 = row['group1'].lower().split(':')[0].replace(',','').replace('ja','')
            # group1 = ' '.join(group1.split())
            train_data.append(gensim.utils.simple_preprocess("{} {}".format(
                row['Nimi'], 
                row['group1'].split(':')[0])))

        model = gensim.models.Word2Vec(train_data, size=150, window=6, min_count=1, workers=multiprocessing.cpu_count(), iter=2000)
        #model.train(train_data, total_examples=len(train_data), epochs=2000)
        self.wv = model.wv

    def _approximate_class(self, word=''):
        highest = {'ingredient': word,'class name': 'None','class id': -1, 'score':0}
        i=0
        if len(word) > 0 and self.wv is not None:
            for c_list in self.preprocessed_classes:
                for c in c_list:
                    score = None
                    try:
                        score = self.wv.similarity(word,c)
                    except Exception as e:
                        #print(e)
                        pass

                    score = 0 if score == None else score
                    
                    if score > highest['score']:
                        highest['score']=score
                        highest['class name']=self.classes_name[i]
                        highest['class id']=self.classes_id[i]
                i = i+1

        return highest

    def predict(self, df_data):
        data = []
        for i, row in df_data.iterrows():
            highest = {
                'real class': row['group1'], 'full name': row['Nimi'], 'dominating name': 'None',
                'predicted class name': 'None','predicted class id': -1, 'score':0}

            for temp in gensim.utils.simple_preprocess(row['Nimi']):#row['Nimi'].lower().split(' '):    
                result = self._approximate_class(temp)
                if result['score'] > highest['score']:
                    highest['dominating name'] = temp
                    highest['predicted class name'] = result['class name']
                    highest['predicted class id'] = result['class id']
                    highest['score'] = result['score']

            data.append(np.array([
                int(highest['real class'].split(':')[1]), 
                highest['predicted class id']]))
        data = pd.DataFrame(data)
        
        return {'score':accuracy_score(data[0], data[1]), 
            'pred': data[1], 'real': data[0]}
