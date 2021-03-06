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
                'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'
                ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# The outlier I need to remove is the total - clearly not a person. I will check later for more
data_dict.pop("TOTAL")

print "I have removed the outlier"

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

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

print "I have imported modules"

## I've chosen 1000 splits because that's what the tester uses
#sss = StratifiedShuffleSplit(n_splits = 1000, random_state = 42)

# Now I need to actually use the sss to split the data into test and train. For that, though, I will just use the tester function

# For the pipeline I am trying SVC here, and a decision tree/Naive Bayes in the commented out estimators
# I have included a Min Max Scaler and a PCA
estimators = [("Scale", MinMaxScaler()), ("Reduce_Dimension", PCA(svd_solver = "randomized", n_components = 11)), ("Classifier", RandomForestClassifier(min_samples_split = 5, criterion = "gini"))]

#estimators = [("Scale", MinMaxScaler()), ("Reduce_Dimension", PCA(svd_solver = "randomized")), ("Classifier", DecisionTreeClassifier())]

#estimators = [("Scale", MinMaxScaler()), ("Reduce_Dimension", PCA(svd_solver = "randomized")), ("Classifier", GaussianNB())]

# Create the pipeline
clf = Pipeline(estimators)

from tester import test_classifier

test_classifier(clf, my_dataset, features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


"""

from draw_features import Draw2D, Draw3D

features_list_three = ['poi',
                # Financial features
                #'salary', #'deferral_payments',
                #'total_payments',
                #, 'loan_advances',
                #'bonus'
                #,'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                # Email features
                'to_messages',
                'from_poi_to_this_person',
                #'from_messages',
                'from_this_person_to_poi'
                #,'shared_receipt_with_poi'
                ]

data_three = featureFormat(my_dataset, features_list_three, sort_keys = True)
labels, features_three = targetFeatureSplit(data_three)

Draw3D(features_three, labels, True, "image3D.png", features_list_three[1], features_list_three[2], features_list_three[3])
"""

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)