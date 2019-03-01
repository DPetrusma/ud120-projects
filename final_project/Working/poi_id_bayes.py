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
                'salary',
                #'deferral_payments',
                'total_payments',
                #'loan_advances',
                'bonus',
                #'restricted_stock_deferred',
                'deferred_income',
                'total_stock_value',
                'expenses',
                'exercised_stock_options',
                'other',
                'long_term_incentive',
                'restricted_stock',
                #'director_fees',
                # Email features
                'to_messages',
                'from_poi_to_this_person',
                'from_messages',
                'from_this_person_to_poi',
                'shared_receipt_with_poi',
                # New features
                'salary_proportion_of_total_payments'
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

for person in my_dataset:
    my_dataset[person]["salary_proportion_of_total_payments"] = my_dataset[person]["salary"] / my_dataset[person]["total_payments"]

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

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

print "I have imported modules"

# I've chosen 1000 splits for this classifier because it can finish on my machine. The other parameters match the tester function
sss = StratifiedShuffleSplit(n_splits = 1000, test_size = 0.1, random_state = 42)

# For the pipeline I am trying Naive Bayes here, and I have included a Min Max Scaler and a PCA
estimators = [("Scale", MinMaxScaler()), ("Reduce_Dimension", PCA(svd_solver = "randomized")), ("Classifier", GaussianNB())]

# Create the pipeline
pipe = Pipeline(estimators)

# These are the parameters to loop through for the different classifiers
params = {  "Reduce_Dimension__n_components" : range(1, 16)
        }
      
# Now, run the grid search to find the best combination of parameters to maximise the f1 score
# Since I include the stratified shuffle split in the grid search, I don't need "features_test" and "features_train"
print "I have set up the Grid Search"

clf_gs = GridSearchCV(pipe, params, cv = sss, scoring = "f1")
clf_gs.fit(features, labels)

print "I have fit the grid search"

clf_gs_pred = clf_gs.predict(features)

# Print out the best parameters so that I can use the best one for clf
print "Best params: ", clf_gs.best_params_

# Give the detailed breakdown of the scores
print "Classification Report: "
print classification_report(y_true = labels, y_pred = clf_gs_pred, target_names = ["Not POI", "POI"])