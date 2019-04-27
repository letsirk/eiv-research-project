#https://www.geeksforgeeks.org/performing-google-search-using-python-code/
#https://dev.to/pranay749254/build-a-simple-python-web-crawler

import requests
from bs4 import BeautifulSoup
#from googlesearch import search 
import pandas as pd
import urlfetch

def readFile():
    file = pd.read_csv('code/distinct-ingredients.txt', sep=';', header=None, encoding = 'utf-8')
    file.columns=['ingredient']
    return file.ingredient.values

def crawlData(ingredient):
    # to search 
    query = 'https://www.google.com/search?q=foodie+'+ingredient.replace(' ','+')

    # Get the target url using google search
    def search(queryUrl):
        webUrl = ''
  
        code = requests.get(queryUrl) #urlfetch.get(queryUrl+"&num=5")
        plain = code.text
        s = BeautifulSoup(plain, "html.parser")

        # Find results, take the one which contains word foodie and entry
        for div in s.findAll('a'):
            result=div['href']
            if '/entry/' in result and 'foodie' in result:
               webUrl = result.replace('/url?q=','').split('&')[0]
               break
        return webUrl

    # Get the content of the target ingredient
    def getContent(url):
        group = {}
        data = {'_Unit':''}
        if url != '':
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
                        if len(nutritionName) <= 20 and nutritionName != 'Tyydyttyneitä' and nutritionName != 'Laktoosi':
                            nutritionValue = div2.contents[0]
                            # Fix nutrition value
                            if 'g' in nutritionValue:
                                nutritionValue=float(nutritionValue.split('g')[0].strip().replace(',','.'))
                                data['_Unit']='g'
                            elif 'ml' in nutritionValue:
                                nutritionValue=float(nutritionValue.split('ml')[0].strip().replace(',','.'))
                                data['_Unit']='ml'
                            elif 'kJ' in nutritionValue:
                                nutritionValue=float(nutritionValue.split('kJ')[0].strip().replace(',','.')) #Energy kJ/kcal
                            elif 'GRM' in nutritionValue: #Salt measure: gram
                                nutritionValue=float(nutritionValue.split('GRM')[0].strip().replace(',','.'))
                            elif 'MGM' in nutritionValue: #Salt measure: milligram
                                nutritionValue=float(nutritionValue.split('MGM')[0].strip().replace(',','.'))/1000
                            elif 'ME' in nutritionValue: #Salt measure: unknown but has to be small cause it occured in basil
                                nutritionValue=0
                            # Fix nutrition name
                            if nutritionName == 'Hiilihydraatti':
                                nutritionName = 'Hiilihydraattia'
                            elif nutritionName == 'Suolaprosentti':
                                nutritionName = 'Suola'
                            elif nutritionName == 'Kuitua':
                                nutritionName = 'Ravintokuitua'

                            data[nutritionName]=nutritionValue
                    else:
                        nutritionName=div2.contents[0]

        return group, data

    searchResult = search(query)
    print(searchResult)
    group, nutritions = getContent(searchResult)
    print(group,nutritions)
    return group, nutritions

# Get distinct ingredients from file
ingredients = readFile()

df_data = pd.DataFrame()
df_data['Nimi'] = ''
df_data_fail = pd.DataFrame()
df_data_fail['Nimi'] = ''

# Extend ingredient inormation with nutrient information
for ingredient in ingredients:
    print(ingredient)

    # Crawl data
    group,nutritions=crawlData(ingredient)
    
    # Store data to dataframe
    if len(group)>0:
        for g in group:
            if g not in df_data.columns:
                df_data[g] = ''
                df_data_fail[g] = ''
    if len(nutritions)>0:
        for n in nutritions:
            if n not in df_data.columns: 
                df_data[n] = '' if n == '_Unit' else 0
                df_data_fail[n] = '' if n == '_Unit' else 0

    if len(nutritions['_Unit'])>0:
        df_data=df_data.append({'Nimi':ingredient,**group,**nutritions}, ignore_index=True)
    else:
        df_data_fail=df_data_fail.append({'Nimi':ingredient,**group,**nutritions}, ignore_index=True)

df_data.to_csv('s-market-ingredients.csv',index=False)
df_data_fail.to_csv('s-market-failed-ingredients.csv',index=False)