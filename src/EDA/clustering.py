from gensim.models import Word2Vec
import nltk
import numpy as np
import pandas as pd
import re
from gensim.models import Word2Vec
from scipy.stats.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle


def read_file(file):
    header_types = {'id': str, 'original': str, 'edit': str, 'grades': str, 'meanGrade':np.float64}
    return pd.read_csv("data/task-1/train.csv", delimiter=",", dtype=header_types).values

def get_words_representation_with_score(data):
    edits = data[:,2]
    edits_array = []
    for edit in edits:
        edits_array.append([edit])
    """
    encodings = Word2Vec(edits_array, size=50, min_count=1, sg=1)
    edit_words_rep = []
    for i in range(edits.shape[0]):
        edit_words_rep.append(encodings.wv[edits[i]])
    edit_words_rep = np.asarray(edit_words_rep)
    """

    original = data[:,1]
    to_replace = []
    for sent in original:
        entity = re.finditer(r'\<.*?\>', sent)
        for item in entity:
            to_replace.append(np.array([item.group(0).replace("<", "").replace("/>", "")]))
    to_replace = np.array(to_replace)

    all_words = np.concatenate((to_replace, edits_array), axis=0)
    encodings = Word2Vec(all_words, size=50, min_count=1, sg=1)
    
    edit_words_rep = []
    orig_words_rep = []
    for i in range(len(to_replace)):
        orig_words_rep.append(encodings.wv[to_replace[i][0]])
        edit_words_rep.append(encodings.wv[edits[i]])
    orig_words_rep = np.asarray(orig_words_rep)
    edit_words_rep = np.asarray(edit_words_rep)


    return orig_words_rep, edit_words_rep, np.array(data[:,-1])

def compute_correlation(original, edited, scores):
    dist = []
    for i in range(len(edited)):
        dist.append(np.linalg.norm(original[i]-edited[i]))
    dist = np.array(dist)

    print(original[0])
    print(edited[0])

    print(dist[0:5])
    print(scores[0:5])

    
    print("Corr coeff: ", pearsonr(dist, scores))

def read_file_clean(file):
    header_types = {'id': str, 'original': str, 'edit': str, 'grades': str, 'meanGrade':np.float64}
    data = pd.read_csv("data/task-1/train.csv", delimiter=",", dtype=header_types)
    #data.original = data.original.map(lambda x: x.replace('<', '').replace('/>',''))
    original = []
    for sentence in data.original:
        orig_word = re.finditer(r'\<.*?\>', sentence)
        for item in orig_word:
            original.append(item.group(0))
            #to_replace.append(item.group(0).replace("<", "").replace("/>", ""))
    for idx in range(len(data)):
        data.iat[int(idx), 1] = data.iat[int(idx), 1].replace(original[idx], data.iat[int(idx),2], True)
        
    data = data.values

    return data

def sentences_to_array(data, tokenized):
    m = Word2Vec(tokenized, size=50, min_count=1, sg=1)
    corpus = []
    mappings = []
    for i, sent in enumerate(tokenized):
        #print(sent)
        ss = np.zeros(50)
        for token in sent:
            rep = m.wv[token]
            ss = ss + rep
        ss = ss / len(sent)
        corpus.append(ss)
        mappings.append((corpus, data[i][0]))
    corpus = np.asarray(corpus)
    mappings = np.asarray(mappings)
    return corpus



if __name__ == "__main__":
    """
    data = read_file("data/task-1/train.csv")
    original, edited, scores = get_words_representation_with_score(data)
    compute_correlation(original, edited, scores)
    """
    
    data = read_file_clean("data/task-1/train.csv")
    tokenized = []
    for sent in data[:,1]:
        tokenized.append(nltk.word_tokenize(sent))

    corpus = sentences_to_array(data[:,0],tokenized)
    """
    wcss = []
    print("Running k-means")
    for i in range(1, 21):
        print("\t Testing with "+str(i)+" clusters.")
        kmeans = KMeans(n_clusters=i, init = 'k-means++')
        kmeans.fit(corpus)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1,21), wcss)
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.savefig("Elbow method")
    #plt.show()
    """

    kmeans = KMeans(3, init='k-means++').fit(corpus)

    for i in range(len(data)):
        np.concatenate((data[i], [kmeans.labels_[i]]))
    
    for c in range(3):
        i = 0
        for element in data:
            if element[-1] == c:
                print(element[1], element[-1])
                i = i+1
            if i == 5:
                break
    
    
    

    


"""
sentence = [
        ['this is the learnig good deep good book'],
        ['this is another book'],
        ['one more book'],
        ['train railway station'],
        ['time train station'],
        ['time railway station train'],
        ['this is the new post'],
        ['this is abbout more deep learning post'],
        ['and this is the one']
        ]
"""
"""
tokens = []
for sent in sentence:
    tokens.append(nltk.word_tokenize(sent[0]))

m = Word2Vec(tokens, size=50, min_count=1, sg=1)

corpus = []
for sent in tokens:
    ss = np.zeros(50)
    for token in sent:
        rep = m.wv[token]
        ss = ss + rep
    
    ss = ss / len(sent)
    corpus.append(ss)

corpus = np.asarray(corpus)
print(corpus.shape)

wcss = []
for i in range(1, 4):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', random_state = 42)
    kmeans.fit(corpus)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,4), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

pickle.dump(kmeans, open("kmeans_model.pkl"), "wb")
"""