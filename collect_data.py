import pandas
import pymongo
import re
import requests
import wikipedia

from argparse import ArgumentParser
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer 

parser = ArgumentParser()
parser.add_argument('-q', '--query', type=str, help='Text to query')


db = pymongo.MongoClient(host='54.187.166.119')
my_wiki_collection = db.my_wiki_database.my_wiki_collection
my_wiki_clean_collection = db.my_wiki_database.my_wiki_clean_collection
cursor = my_wiki_collection.find()
cursor_clean = my_wiki_clean_collection.find()




def request(CATEGORY):
    '''
    Queries Wikipedia API for a list of articles from input CATEGORY
    '''
    
    category = re.sub(' ', '+',CATEGORY).lower()
    response = requests.get(

'https://en.wikipedia.org/w/api.php?action=query&format=json&list=categorymembers&cmtitle=Category%3A+{}&cmlimit=max'.format(category)
                            )
    
    return(response.json())


def make_df(category):
    '''
    Makes a DataFrame of 'pageid' and 'titles' from input category
    '''
    df = pandas.DataFrame(request(category)['query']['categorymembers'])
    return(df)



def throw_articles_in_mongo(category, n_levels=2):
    '''
    Stores articles under category in Mongo
    
    n_levels = number of subcategories to pull articles
    '''

    for i, title in enumerate(make_df(category)['title']):
        try:
            dump_in_mongo={}
            if 'Category:' not in title:
                pageid=make_df(category).iloc[i]['pageid'],
                title= make_df(category).iloc[i]['title'],
                content= wikipedia.page(pageid=pageid).content,
                sub_category= category,
                dump_in_mongo['Pageid']= str(pageid)
                dump_in_mongo['Title']= title
                dump_in_mongo['Content']= content
                dump_in_mongo['Subcategory']= sub_category           

                my_wiki_collection.insert_one(dump_in_mongo)
                
                
            elif 'Category:' in title:
                  while n_levels>0:
                    sub_cat=title.split(':')[1]
                    throw_in_mongo(sub_cat, n_levels -1)
                    break
        except:
                pass   


def clean_column_tuple(column):
    column= re.sub('[\D]','', column)
    return column


def clean_column_list(column):
    column=column[0]
    return column


def clean_df(df, category):
    '''
    Cleans dataframe 
    '''
    clean_me=df
    clean_me.drop('_id', axis=1, inplace=True)
    for columns in df.columns:
        if type(clean_me[columns][0])==list:
            clean_me[columns]=clean_me[columns].apply(clean_column_list)
        
        elif type(clean_me[columns][0])==str:
            clean_me[columns]=clean_me[columns].apply(clean_column_tuple)
    clean_me.drop_duplicates(inplace=True)
    clean_me['Category']=category
    return clean_me

def clean_docs_to_mongo(category, n_levels=2):

    '''
    Grabs articles from wikipedia, cleans it, and puts it in mongo 
    '''    
        
    throw_in_mongo(category,n_levels)

    list_of_dict = []
    for i in range(my_wiki_collection.count()):
        list_of_dict.append(cursor.next())

    df_dirty = pandas.DataFrame(list_of_dict)

    df_clean = clean_df(df_dirty, category)

    df_to_dict = df_clean.to_dict(orient='records')

    list_of_dicts = []
    for dicts in df_to_dict:
        my_wiki_clean_collection.insert_one(dicts)

    my_wiki_collection.delete_many({})


def get_clean_df():    
    '''
    Creates a clean dataframe of all categories 
    '''
    
    list_of_dicts= []
    for i in range(my_wiki_clean_collection.count()):
        list_of_dicts.append(cursor_clean.next())

    category_df=pandas.DataFrame(list_of_dicts)
    category_df.drop('_id', axis=1, inplace=True)
    
    
  
    return category_df
              
def search_term(term):
    '''
    Finds the top 5 related articles
    '''
    category_df = get_clean_df()
    
    tfidf_vectorizer= TfidfVectorizer(min_df=1, stop_words='english')
    document_term_matrix= tfidf_vectorizer.fit_transform(category_df['Content'])
                                
    SVD= TruncatedSVD(125)
    lsa= SVD.fit_transform(document_term_matrix)   
    
    search_term_vec = tfidf_vectorizer.transform([term])
    search_term_lsa = SVD.transform(search_term_vec)
    cosine_similarities = lsa.dot(search_term_lsa.T).ravel()
    results = cosine_similarities.argsort()[:-6:-1]
    
    list_of_rec=[]
    for result in results:
        list_of_rec.append(category_df.ix[result])
    return list_of_rec

#args = parser.parse_args()

#print(search_term(parser.parse_args()))             
              

