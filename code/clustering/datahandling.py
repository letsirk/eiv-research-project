import pandas as pd
from gensim.corpora import WikiCorpus
from gensim.test.utils import datapath
import time
import os

def load(filename='s-market-ingredients.csv'):
    df_data = pd.read_csv('s-market-ingredients.csv')

    #print(df_data.columns)
    #print(df_data.group1.unique())
    #print(df_data[df_data['group1']=='Hedelmät ja vihannekset:1'].group1.unique())
    #df_data = df_data[df_data['group1']=='Hedelmät ja vihannekset:1']
    group1_columns = [
        'Maito:351','Juomat:994','Juustot:417','Valmisruoka:497','Hedelmät ja vihannekset:1',
        'Leipä:701','Liha:80','Lemmikit:1337','Makeiset, jäätelöt ja naposteltavat:1144',
        'Kuivatuotteet:895','Leivonta ja maustaminen:807','Munat:1784','Pakasteet:1075',
        'Rasvat ja öljyt:476','Kala:262','Terveystuotteet:12129']
    # group1_columns = [
    #     'Maito:351','Juomat:994','Juustot:417','Valmisruoka:497','Hedelmät ja vihannekset:1',
    #     'Leipä:701','Liha:80','Makeiset, jäätelöt ja naposteltavat:1144',
    #     'Leivonta ja maustaminen:807','Munat:1784','Pakasteet:1075',
    #     'Rasvat ja öljyt:476','Kala:262','Terveystuotteet:12129']
    
    df_data = df_data[df_data['group1'].isin(group1_columns)]
    #df_data = df_data[df_data['Nimi'].str.contains('tomaatti')]
    #df_data = df_data[['Nimi','Rasvaa','Hiilihydraattia','Proteiinia','Ravintokuitua']] #,'Energiaa','Suola','Sokeria']]
    df_data=df_data.fillna(0)
    return df_data

'''
Creates a corpus from Wikipedia dump file.
(Latest dump files: https://dumps.wikimedia.org/fiwiki/latest/)

Copied from https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html
and from https://github.com/panyang/Wikipedia_Word2vec 
'''
class WikiDumpToCorpus():
    def __init__(self, dump_filename):
        self.dump_file = dump_filename #datapath(dump_filename)
        self.wiki_file = 'wiki_fi.txt'
        self.corpus_file = None

    def make_corpus(self):
        """Convert Wikipedia xml dump file to text corpus"""

        output = open(self.wiki_file, 'w',encoding="utf-8")
        wiki = WikiCorpus(self.dump_file)
        i = 0
        for text in wiki.get_texts():
            output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
            i = i + 1
            if (i % 10000 == 0):
                print('Processed ' + str(i) + ' articles')
            
        output.close()
        print('Processing complete!')

    def open_corpus(self, func):
        """Open corpus file and execute given function for each line"""
        with open(self.wiki_file,'r',encoding="utf-8") as openfileobject:
            func(openfileobject.read())

    def check_corpus(self):
        """Check if corpus file exists"""
        return os.path.exists(self.wiki_file)

