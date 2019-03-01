#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
features_list = ['poi',
                # Financial features
                'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                # Email features
                #'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'
                ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
clf = SVC(gamma = 0.1, C = 1000)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
# Some code for scaling
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
features_train_scaled = min_max_scaler.fit_transform(features_train)
features_test_scaled = min_max_scaler.transform(features_test)
    
from sklearn.decomposition import PCA
    
# Grab the top 10 principal components
#pca = RandomizedPCA(n_components = 10, whiten=True).fit(features_train)
#pca = PCA(svd_solver = "randomized", n_components = 3).fit(features_train)
pca = PCA(svd_solver = "randomized", n_components = 3).fit(features_train_scaled)

features_train_pca = pca.transform(features_train_scaled)
features_test_pca = pca.transform(features_test_scaled)
"""
print "Variance explained by best n PCA:"
for r in  pca.explained_variance_ratio_:
    if r >= 0.1:
        print r
#print pca.components_
"""
# Quick and dirty to get some output
"""
clf.fit(features_train, labels_train)
print "Score: ", clf.score(features_test, labels_test)
"""
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# This is the grid search to test many parameters for an algorithm
param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf_pca_svm = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, scoring = "f1")

from sklearn import metrics
#from sklearn import cross_validation
from sklearn import model_selection
"""
# These are both zero since no predictions of POI were made
print "Precision: ", metrics.precision_score(labels_test, clf.predict(features_test))
print "Recall: ", metrics.recall_score(labels_test, clf.predict(features_test))
print "F1 Score: ", metrics.f1_score(labels_test, clf.predict(features_test))
"""
clf.fit(features_train_pca, labels_train)
print "Score with PCA: ", clf.score(features_test_pca, labels_test)
"""
print "Precision with PCA: ", metrics.precision_score(labels_test, clf.predict(features_test_pca))
print "Recall with PCA: ", metrics.recall_score(labels_test, clf.predict(features_test_pca))
print "F1 Score with PCA: ", metrics.f1_score(labels_test, clf.predict(features_test_pca))

clf_pca_svm.fit(features_train_pca, labels_train)
print "Score with PCA and grid: ", clf_pca_svm.score(features_test_pca, labels_test)
# The below gives me the scores across 5 stratified divisions of the train and test data
print "Score with PCA and grid: ", model_selection.cross_val_score(clf_pca_svm, features_test_pca, labels_test, cv = 5, scoring = "accuracy")
"""
#print "Precision with PCA: ", metrics.precision_score(labels_test, clf_pca_svm.predict(features_test_pca))
#print "Recall with PCA: ", metrics.recall_score(labels_test, clf_pca_svm.predict(features_test_pca))
#print "F1 Score with PCA: ", metrics.f1_score(labels_test, clf_pca_svm.predict(features_test_pca))

# Some pipeline testing
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit

cv = StratifiedShuffleSplit(n_splits = 3, test_size = 0.3, random_state = 42)
estimators = [("Scale", MinMaxScaler()), ("Reduce_Dimension", PCA(svd_solver = "randomized")), ("Classifier", SVC())]
#estimators = [("Scale", MinMaxScaler()), ("Reduce_Dimension", PCA(svd_solver = "randomized")), ("Classifier", DecisionTreeClassifier())]
pipe = Pipeline(estimators)
params = {  "Reduce_Dimension__n_components" : [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]
            #,"Classifier__min_samples_split" : [2,10,40]
            #,"Classifier__criterion" : ["gini", "entropy"]
            ,"Classifier__C" : [1e3, 5e3, 1e4, 5e4, 1e5]
            ,"Classifier__gamma" : [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
            #,"Classifier__kernel" : ["rbf", "linear"]
        }
        
gs_test = GridSearchCV(pipe, params, cv = cv, scoring = "f1")
gs_test.fit(features_train, labels_train)
gs_test_pred = gs_test.predict(features_test)

#print "Best estimator: ", gs_test.best_estimator_
print "Best params: ", gs_test.best_params_
#print "Scorer: ", gs_test.scorer_
#print "Splits: ", gs_test.n_splits_

print "Classification Report: "
print metrics.classification_report(y_true = labels_test, y_pred = gs_test_pred, target_names = ["Not POI", "POI"])
#cv = 5 means a stratified 5-fold cross validation
scores = model_selection.cross_val_score(gs_test, features_test, labels_test, cv = cv, scoring = "precision_weighted")
print "Weighted Average Precision with Pipeline: ", scores
print("Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = model_selection.cross_val_score(gs_test, features_test, labels_test, cv = cv, scoring = "recall_weighted")
print "Weighted Average Recall with Pipeline: ", scores
print("Recall: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from tester import test_classifier

test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)