import numpy as np
from sklearn.manifold import TSNE
import pickle

import matplotlib.pyplot as plt


def transform_model1(W):
    a=np.load('a.npy') #this will be column vector
    W_new = W*np.expand_dims(a,axis=1).T
    return W_new

def transfrom_model2(W):
    a=np.load() #this will be column vector
    s=np.load()
    
    W_a = W*np.expand_dims(a,axis=1).T
    W_s = W*np.expand_dims(s,axis=1).T
    
    return W_a,W_s

def transform_model3(W):
    A= np.load('a_model3.npy')
    A=A.T
    return np.matmul(W,A)


    
word_list=pickle.load(open("word_list.pkl", "rb"))
X=pickle.load(open("embedding_matrix.pkl", "rb"))

word_pair = pickle.load(open("word_pairs.pkl", "rb"))
embd_pairs = pickle.load(open("pairs.pkl", "rb"))


np.save('glove_X_embedding.npy',TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X))
np.save('model1_X_embedding.npy',TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(transform_model1(X)))
np.save('model3_X_embedding.npy',TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(transform_model3(X)))


X_embedded_1 = np.load('model1_X_embedding.npy')
X_embedded_3 = np.load('model3_X_embedding.npy')
X_embedded_glove = np.load('glove_X_embedding.npy')

X_embedded_list = [X_embedded_glove, X_embedded_1,X_embedded_3]

indices = [28, 110, 239, 781, 1023, 985, 1056, 1090, 1230, 1839] #Indices of pairs to plot

for X_embedded in X_embedded_list:

    select_word_pairs = [word_pair[i][0] for i in indices]+ [word_pair[i][1] for i in indices]
    select_word_indices = [word_list.index(word) for word in select_word_pairs]
    select_word_embeddings = np.array([X_embedded[i,:] for i in select_word_indices])
    
    plt.scatter(select_word_embeddings[:,0],select_word_embeddings[:,1],linewidths=1,color='blue')
    plt.grid()
    plt.xlabel("X1",size=15)
    plt.ylabel("X2",size=15)
    plt.title("Word Embedding Space",size=20)
    
    for i, word in enumerate(select_word_pairs):
        plt.annotate(word,xy=(select_word_embeddings[i,0],select_word_embeddings[i,1]))
    for i in range(select_word_embeddings.shape[0]):
        plt.arrow(0,0,select_word_embeddings[i,0],select_word_embeddings[i,1])
    
    plt.show()




