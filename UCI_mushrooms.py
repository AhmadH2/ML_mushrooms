
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

"""
he data used in this notebook is based on the UCI Mushroom Data Set stored in mushrooms.csv.
http://archive.ics.uci.edu/ml/datasets/Mushroom?ref=datanews.io
In order to better vizualize the decision boundaries, we'll perform Principal Component Analysis (PCA)
on the data to reduce the dimensionality to 2 dimensions. Dimensionality reduction.
Play around with different models and parameters to see how they affect the classifier's decision boundary and accuracy!
"""

df = pd.read_csv('mushrooms.csv')
mushroom_dataset = pd.get_dummies(df)
mushroom_dataset_sample = mushroom_dataset.sample(frac=0.1)

X = mushroom_dataset_sample.iloc[:,2:]
y = mushroom_dataset_sample.iloc[:,1]

pca = PCA(n_components=2).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(pca, y, random_state=0)

plt.figure(dpi=120)
plt.scatter(pca[y.values==0,0], pca[y.values==0,1], alpha=0.5, label='Edible', s=2)
plt.scatter(pca[y.values==1,0], pca[y.values==1,1], alpha=0.5, label='Poisonous', s=2)
plt.legend()
plt.title('Mushroom Data Set\nFirst Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.gca().set_aspect('equal')
plt.show()


def plot_mushroom(X, y, model):

    plt.figure()
    mesh_step_size = 0.01  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), 
                         np.arange(y_min, y_max, mesh_step_size))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.scatter(X[y.values==0,0], X[y.values==0,1], alpha=0.4, label='Edible', s=5)
    plt.scatter(X[y.values==1,0], X[y.values==1,1], alpha=0.4, label='Posionous', s=5)
    plt.imshow(Z, interpolation='nearest', cmap='RdYlBu_r', alpha=0.15,
               extent=(x_min, x_max, y_min, y_max), origin='lower')
    plt.legend()
    plt.title('Decision Boundary\n' + str(model).split('(')[0] + ' Test Accuracy: ' 
              + str(np.round(model.score(X, y), 5)))
    
    plt.gca().set_aspect('equal');


#support vector classifier model
svc_model = SVC(kernel='rbf',C=10, gamma='auto')
svc_model.fit(X_train,y_train)

plot_mushroom(X_test, y_test, svc_model)


#Decision Tree Classifier model
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train,y_train)
plot_mushroom(X_test, y_test, DT_model)


#Random Forest Classifier model
RFC_model = RandomForestClassifier()
RFC_model.fit(X_train,y_train)

plot_mushroom(X_test, y_test, RFC_model)