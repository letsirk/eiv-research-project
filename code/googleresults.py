#https://www.geeksforgeeks.org/performing-google-search-using-python-code/
#https://dev.to/pranay749254/build-a-simple-python-web-crawler

import requests
from bs4 import BeautifulSoup
from googlesearch import search 
import pandas as pd

def readFile():
    file = pd.read_csv('distinct-ingredients.txt', sep=';', header=None, encoding = 'utf-8')
    file.columns=['ingredient']
    return file.ingredient.values

def crawlData(ingredient):
    # to search 
    query = ingredient+" foodie.fi"

    def web(page,WebUrl):
        group = {}
        data = {'_Unit':''}
        if(page>0):
            url = WebUrl
            code = requests.get(url)
            plain = code.text
            s = BeautifulSoup(plain, "html.parser")

            # Find grouping
            i=0
            for div in s.findAll('a', {'class':'js-category-item'}):
                productId=div['href']
                if '/products/' in productId:
                    i += 1
                    productId=productId.replace('/products/','')
                    group['group'+str(i)]=div.contents[0].strip()+':'+productId

            # Find nutritions
            i=0
            for div in s.findAll('div',{'class':'tab-pane', 'id':'nutritions'}):
                nutritionName = ''
                for div2 in div.find('table').findAll('td'):
                    i += 1
                    if i % 3 == 0: # do nothing
                        i=0
                    elif i % 2 == 0: # g/ml
                        if len(nutritionName) <= 20:
                            nutritionValue = div2.contents[0]
                            if 'g' in nutritionValue:
                                nutritionValue=float(nutritionValue.split('g')[0].strip().replace(',','.'))
                                data['_Unit']='g'
                            elif 'ml' in nutritionValue:
                                nutritionValue=float(nutritionValue.split('ml')[0].strip().replace(',','.'))
                                data['_Unit']='ml'
                            elif 'kJ' in nutritionValue:
                                nutritionValue=float(nutritionValue.split('kJ')[0].strip().replace(',','.')) #Energy kJ/kcal
                            data[nutritionName]=nutritionValue
                    else:
                        nutritionName=div2.contents[0]

        return group, data

    group = {}
    nutritions ={}
    for j in search(query, tld="fi", num=1, stop=1, pause=2): 
        print(j)
        if '/entry/' in j:
            group, nutritions = web(1,j)
        print(group,nutritions)
    return group, nutritions

ingredients = readFile()
for ingredient in ingredients:
    print(ingredient)
    group,nutritions=crawlData(ingredient)



