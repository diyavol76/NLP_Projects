import yaml
import csv
import pandas as pd
import os
import json
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances

def get_csv_data():
    with open('config.yaml') as fil:
        config = yaml.load(stream=fil, Loader=yaml.FullLoader)
        csv_path=os.path.join(config['DATA']['DATA_PATH'],config['DATA']['DATA_NAME'])
        data=pd.read_csv(csv_path)
    return data

def genres_and_keywords_to_string(row):
    genres = json.loads(row['genres'])
    genres = ' '.join(''.join(j['name'].split()) for j in genres)

    keywords = json.loads(row['keywords'])
    keywords = ' '.join(''.join(j['name'].split()) for j in keywords)
    return "%s %s" % (genres,keywords)

def test_functions():

    df=get_csv_data()
    #print(df)

    print(df.head())
    x=df.iloc[0]
    print(x)
    print(x['genres'])
    print(x['keywords'])

    j= json.loads(x['genres'])
    print(j)
    print(' '.join(''.join(jj['name'].split()) for jj in j))



    #create a new string representation of each movie
    df['string'] = df.apply(genres_and_keywords_to_string, axis=1)

    tfidf = TfidfVectorizer(max_features=2000)

    X=tfidf.fit_transform(df['string'])

    print(X)

    movie2idx = pd.Series(df.index, index=df['title'])
    print(movie2idx)

    idx= movie2idx['The Prestige']
    print(idx)

    query= X[idx]
    print(query)
    print(query.toarray())

    scores=cosine_similarity(query,X)
    print(scores)
    scores=scores.flatten()

    plt.plot(scores)
    plt.figure()
    print((-scores).argsort())
    plt.plot(scores[(-scores).argsort()])
    plt.show()

    recommend_idx = (-scores).argsort()[1:6]

    print(df['title'].iloc[recommend_idx])

#create a function
def recommend(df,movie2idx,X,title):
    #get the row in the dataframe for this movie
    idx = movie2idx[title]
    if type(idx) == pd.Series:
        idx = idx.iloc(0)

    #calculate the pairwise similarities for movie
    query = X[idx]
    scores = cosine_similarity(query,X)

    #1xN to 1-D array
    scores=scores.flatten()

    recommend_idx = (-scores).argsort()[1:6]

    return df['title'].iloc[recommend_idx]


if __name__ == '__main__':
    df = get_csv_data()

    df['string'] = df.apply(genres_and_keywords_to_string, axis=1)

    tfidf = TfidfVectorizer(max_features=2000)

    XX = tfidf.fit_transform(df['string'])

    movie2idx = pd.Series(df.index, index=df['title'])

    print(recommend(df,movie2idx, XX , 'Mortal Kombat'))

    print(recommend(df,movie2idx, XX , 'The Prestige'))

    print(recommend(df, movie2idx, XX, 'Inception'))


