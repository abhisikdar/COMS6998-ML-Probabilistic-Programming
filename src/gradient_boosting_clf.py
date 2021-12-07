from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.model_selection import cross_val_score
import numpy as np

embd_pairs = pickle.load(open("pairs.pkl", "rb"))
a = np.load('a.npy').T
a_3 = np.load('a_model3.npy').T

X = embd_pairs[:,:200]
y = embd_pairs[:,200]

X_w1 = X[:, :100]
X_w2 = X[:, 100:]

X_w1_transformed_model1 = X_w1*a
X_w2_transformed_model1 = X_w2*a
X_transformed_model1 = np.hstack([X_w1_transformed_model1, X_w2_transformed_model1])

#Training a classifier on raw glove embeddings
clf = GradientBoostingClassifier()
scores = cross_val_score(clf, X, y, cv=5) #cross validation
print("Raw glove embeddings gave %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#Training a classifier on glove embeddings transformed using model 1
clf_model1 = GradientBoostingClassifier()
scores_model1 = cross_val_score(clf_model1, X_transformed_model1, y, cv=5)
print("Model 1 glove embeddings gave %0.2f accuracy with a standard deviation of %0.2f" % (scores_model1.mean(), scores_model1.std()))

X_w1_transformed_model3 = np.matmul(X_w1, a_3)
X_w2_transformed_model3 = np.matmul(X_w2, a_3)
X_transformed_model3 = np.hstack([X_w1_transformed_model3, X_w2_transformed_model3])

#Training a classifier on glove embeddings transformed using model 3
clf_model3 = GradientBoostingClassifier()
scores_model3 = cross_val_score(clf_model3, X_w1_transformed_model3, y, cv=5)
print("Model 3 glove embeddings gave %0.2f accuracy with a standard deviation of %0.2f" % (scores_model3.mean(), scores_model3.std()))
