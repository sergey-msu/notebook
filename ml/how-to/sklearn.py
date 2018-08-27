
# scale matrix columns to mean=0 std=1

from sklearn.preprocessing import scale
X = scale(X)

or
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit with train
X_test_scaled  = scaler.transform(X_test)       # use in test


# Decision tree

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=241)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)


# K-Neighbors

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
kn = KNeighborsClassifier(n_neighbors=1)
score = cross_val_score(kn, X, y, cv=kf, scoring='accuracy').mean()


# SVM

from sklearn.svm import SVC

svm = SVC(C=100000, kernel='linear', random_state=241)
svm.fit(X, y)


# Logistic regression

from sklearn.linear_mo
del import LogisticRegression

lr = LogisticRegression(C=10, max_iter=10000)
lr.fit(X, y)
y_pred = lr.predict_proba(X)[:, 1]


# Ridge (linear with L2 regularization) regression

from sklearn.linear_model import Ridge

lr = Ridge(alpha=1, random_state=241)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# Lasso (linear with L1 regularization) regression

from sklearn.linear_model import Lasso

lr = Lasso(alpha=1, random_state=241)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# Random Forests

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=i, random_state=1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# Bagging - 
#   1. re-train base algorithm on subsamples that are drawn from base sample with replacement, then average the rasults
#   2. train on a feature subsets

alg = DecisionTreeClassifier()
bag = BaggingClassifier(alg, n_estimators=100)
score = cross_val_score(bag, X, y, cv=10, n_jobs=-1)



# Gradient Boosting (with GB learning iterations visualization)

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=250, learning_rate=0.01, random_state=241)
gbc.fit(X_train, y_train)

test_score  = []
train_score = []
for i, y_pred in enumerate(gbc.staged_decision_function(X_test)):
    test_score.append(gbc.loss_(y_test, y_pred))   # may use any other metrics
for i, y_pred in enumerate(gbc.staged_decision_function(X_train)):
    train_score.append(gbc.loss_(y_train, y_pred)) # may use any other metrics

plt.plot(test_score)
plt.plot(train_score)
plt.legend(['test score', 'train score'])
plt.show()


# Word vectorization

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer()
descr_tfidf_train = vect.fit_transform(df_train['FullDescription'])
descr_tfidf_test  = vect.transform(df_test['FullDescription'])


# Meta params grid search

from sklearn.model_selection import GridSearchCV, KFold

param_grid = { 'C': [1, 10, 100, 1000] }
svm = SVC(kernel='linear', random_state=241)
kf = KFold(n_splits=5, shuffle=True, random_state=241)
grid = GridSearchCV(svm, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

grid.best_estimator_
grid.best_score_  # grid.grid_scores_  - scores for all parameters
grid.best_params_


# Randomized grid search - for many parameters

grid = RandomizedSearchCV(alg, param_grid, cv=kf, scoring='accuracy', n_iters=20, random_state=20, n_jobs=-1)


# Different metrics

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

score = accuracy_score(y_test, y_pred)


# PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(df)
tr = pca.transform(df)

pca.explained_variance_ratio_             # explain variance by main components
np.cumsum(pca.explained_variance_ratio_)  # cummulative explanation


# Folding sample  

  # отложенная выборка
  from sklearn.model_selection train_test_split
  
  X_train, X_test, y_train, y_test = 
                     train_test_split(X, y, 
                                      test_size=0.33, 
                                      random_state=84,
                                      stratify=y)  # the last if we want to preserve y-class balance
  
  
  # k-fold
  from sklearn.model_selection import KFold
  
  kf = KFold(n_splits=5, shuffle=True, random_state=84)
  for train_ids, test_ids in kf.split(X):
      X_train, X_test = X[train_ids], X[test_ids]
      y_train, y_test = y[train_ids], y[test_ids]
      # ...

      
  # stratified k-fold (preserving the percentage of samples for each class)
  from sklearn.model_selection import StratifiedKFold
  
  kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=84)
  for train_ids, test_ids in kf.split(X):
      X_train, X_test = X[train_ids], X[test_ids]
      y_train, y_test = y[train_ids], y[test_ids]
      # ...
      
  # ShuffleSplit, StratifiedShuffleSplit - нет ограничения на уникальность элементов выборки, 
  # перемешивает перед каждым разбиением
  from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
  
  # LeaveOneOut
  from sklearn.model_selection import LeaveOneOut
  
  
# Confusion Matrix: a[i, j] = # of obj of i-th class but classified with j-th class

from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, y_pred)


# Precision (точность) and recall (полнота), report

from sklearn.metrics import precision_score, recall_score, classification_report

print(classification_report(y_test, y_pred)) # precision, recall, f1-score, averages


# One Hot Encoding of all categorical features

X_cat_oh = encoder.fit_transform(X_cat.T.to_dict().values())


# Pipeline: before fit model we want to scale our data. Scaler fits on train part of the sample
#           So there is a problem with CV validation: we must apply different scale before each fold step!
#           Therefore we must: 1. walk through CV manually, 2. use automated pipeline

from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps = [('scaling', scaler), ('regression', estimator)])
pipeline.fit(X_train, y_train)


# Pipeline real-world example: split features by types, scale float, one-hot-encode catagorical, join back and train estimator

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

pipeline = Pipeline(steps = [       
    ('feature_processing', pipeline.FeatureUnion(transformer_list = [        
            #binary
            ('binary_variables_processing', FunctionTransformer(lambda data: data[:, binary_data_indices])), 
                    
            #numeric
            ('numeric_variables_processing', Pipeline(steps = [
                ('selecting', FunctionTransformer(lambda data: data[:, numeric_data_indices])),
                ('scaling', StandardScaler(with_mean = 0))            
                        ])),

            #categorical
            ('categorical_variables_processing', Pipeline(steps = [
                ('selecting', FunctionTransformer(lambda data: data[:, categorical_data_indices])),
                ('hot_encoding', OneHotEncoder(handle_unknown = 'ignore'))            
                        ])),
        ])),
    ('model_fitting', estimator)
    ]
)


# Learning Curve - dependense of model quality of training set volume 
# - имеет ли смысл увеличивать объем обучающей выборки для данного алгоритма

from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(alg, X_train, y_train, 
                                                        train_sizes=np.arange(0.1, 1.0, 0.2), 
                                                        cv=3, scoring='accuracy')

plt.grid(True)
plt.plot(train_sizes, train_scores.mean(axis = 1), 'g-', marker='o', label='train')
plt.plot(train_sizes, test_scores.mean(axis = 1), 'r-', marker='o', label='test')
plt.ylim((0.0, 1.05))
plt.legend(loc='lower right')
plt.show()