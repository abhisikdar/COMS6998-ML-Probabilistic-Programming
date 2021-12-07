from sklearn import svm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import spatial
from tsne import *

embd_pairs = pickle.load(open("pairs.pkl", "rb"))

X = embd_pairs[:,:200]
y = embd_pairs[:,200]

X_w1 = X[:, :100]
X_w2 = X[:, 100:]

X_w1_transformed = transform_model1(X_w1)
X_w2_transformed = transform_model1(X_w2)

synonyms = []
antonyms = []
synonyms_transformed = []
antonyms_transformed = []

for i in range(len(y)):
    if y[i]:
        synonyms.append(1 - spatial.distance.cosine(X_w1[i], X_w2[i]))
    else:
        antonyms.append(1 - spatial.distance.cosine(X_w1[i], X_w2[i]))
        
plt.hist(synonyms, bins=40, range=(-1, 1), alpha=0.5, label='synonym_pairs')
plt.xticks(np.arange(-1, 1.25, step=0.25))
plt.hist(antonyms, bins=40, range=(-1, 1), alpha=0.5, label='antonym_pairs')
plt.xticks(np.arange(-1, 1.25, step=0.25))
plt.xlabel("Cosine Similarity")
plt.ylabel("n_pairs")
plt.legend(loc='best')
plt.savefig("raw_glove_embeddings_hist.png")
plt.close()

for i in range(len(y)):
    if y[i]:
        synonyms_transformed.append(1 - spatial.distance.cosine(X_w1_transformed[i], X_w2_transformed[i]))
    else:
        antonyms_transformed.append(1 - spatial.distance.cosine(X_w1_transformed[i], X_w2_transformed[i]))

plt.hist(synonyms_transformed, bins=40, range=(-1, 1), alpha=0.5, label='synonym_pairs(transformed, model1)')
plt.xticks(np.arange(-1, 1.25, step=0.25))
plt.hist(antonyms_transformed, bins=40, range=(-1, 1), alpha=0.5, label='antonym_pairs(transformed, model1)')
plt.xticks(np.arange(-1, 1.25, step=0.25))
plt.xlabel("Cosine Similarity")
plt.ylabel("n_pairs")
plt.legend(loc='best')
plt.savefig("model1_transformed_embeddings_hist.png")
plt.close()

X_w1_transformed_a, X_w1_transformed_s = transfrom_model2(X_w1)
X_w2_transformed_a, X_w2_transformed_s = transfrom_model2(X_w2)

synonyms_transformed = []
antonyms_transformed = []

for i in range(len(y)):
    if y[i]:
        synonyms_transformed.append(1 - spatial.distance.cosine(X_w1_transformed_a[i], X_w2_transformed_a[i]))
    else:
        antonyms_transformed.append(1 - spatial.distance.cosine(X_w1_transformed_a[i], X_w2_transformed_a[i]))

plt.hist(synonyms_transformed, bins=40, range=(-1, 1), alpha=0.5, label='synonym_pairs(transformed, model2_antonym_transformation)')
plt.xticks(np.arange(-1, 1.25, step=0.25))
plt.hist(antonyms_transformed, bins=40, range=(-1, 1), alpha=0.5, label='antonym_pairs(transformed, model2_antonym_transformation)')
plt.xticks(np.arange(-1, 1.25, step=0.25))
plt.xlabel("Cosine Similarity")
plt.ylabel("n_pairs")
plt.legend(loc='best')
plt.savefig("model2_transformed_embeddings_antonyms_hist.png")
plt.close()

synonyms_transformed = []
antonyms_transformed = []

for i in range(len(y)):
    if y[i]:
        synonyms_transformed.append(1 - spatial.distance.cosine(X_w1_transformed_s[i], X_w2_transformed_s[i]))
    else:
        antonyms_transformed.append(1 - spatial.distance.cosine(X_w1_transformed_s[i], X_w2_transformed_s[i]))

plt.hist(synonyms_transformed, bins=40, range=(-1, 1), alpha=0.5, label='synonym_pairs(transformed, model2_synonym_transformation)')
plt.xticks(np.arange(-1, 1.25, step=0.25))
plt.hist(antonyms_transformed, bins=40, range=(-1, 1), alpha=0.5, label='antonym_pairs(transformed, model2_synonym_transformation)')
plt.xticks(np.arange(-1, 1.25, step=0.25))
plt.xlabel("Cosine Similarity")
plt.ylabel("n_pairs")
plt.legend(loc='best')
plt.savefig("model2_transformed_embeddings_synonyms_hist.png")
plt.close()

synonyms_transformed = []
antonyms_transformed = []

for i in range(len(y)):
    if y[i]:
        synonyms_transformed.append(1 - spatial.distance.cosine(X_w1_transformed_s[i], X_w2_transformed_s[i]))
    else:
        antonyms_transformed.append(1 - spatial.distance.cosine(X_w1_transformed_a[i], X_w2_transformed_a[i]))

plt.hist(synonyms_transformed, bins=40, range=(-1, 1), alpha=0.5, label='synonym_pairs(transformed, model2_synonym_transformation)')
plt.xticks(np.arange(-1, 1.25, step=0.25))
plt.hist(antonyms_transformed, bins=40, range=(-1, 1), alpha=0.5, label='antonym_pairs(transformed, model2_antonym_transformation)')
plt.xticks(np.arange(-1, 1.25, step=0.25))
plt.xlabel("Cosine Similarity")
plt.ylabel("n_pairs")
plt.legend(loc='best')
plt.savefig("model2_transformed_embeddings_antonyms_synonyms_hist.png")
plt.close()

X_w1_transformed = transform_model3(X_w1)
X_w2_transformed = transform_model3(X_w2)

synonyms_transformed = []
antonyms_transformed = []

for i in range(len(y)):
    if y[i]:
        synonyms_transformed.append(1 - spatial.distance.cosine(X_w1_transformed[i], X_w2_transformed[i]))
    else:
        antonyms_transformed.append(1 - spatial.distance.cosine(X_w1_transformed[i], X_w2_transformed[i]))

plt.hist(synonyms_transformed, bins=40, range=(-1, 1), alpha=0.5, label='synonym_pairs(transformed, model3)')
plt.xticks(np.arange(-1, 1.25, step=0.25))
plt.hist(antonyms_transformed, bins=40, range=(-1, 1), alpha=0.5, label='antonym_pairs(transformed, model3)')
plt.xticks(np.arange(-1, 1.25, step=0.25))
plt.xlabel("Cosine Similarity")
plt.ylabel("n_pairs")
plt.legend(loc='best')
plt.savefig("model3_transformed_embeddings_hist.png")
plt.close()
