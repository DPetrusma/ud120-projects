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
## I removed these features one by one, running the code each time, to maximise the F1 score from the tester function
features_list = ['poi',
                ## Financial features
                'salary',
                'deferral_payments', #
                'total_payments',
                'loan_advances', #
                'bonus',
                'restricted_stock_deferred', #
                'deferred_income',
                'total_stock_value',
                'expenses', #
                'exercised_stock_options',
                'other', #
                'long_term_incentive',
                'restricted_stock', #
                'director_fees',
                ## Email features
                'to_messages', #
                'from_poi_to_this_person',
                'from_messages', #
                'from_this_person_to_poi'
                'shared_receipt_with_poi', #
                ## New features
                'salary_proportion_of_total_payments' #
                ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# The outlier I need to remove is the total - clearly not a person. I will check later for more
data_dict.pop("TOTAL")
# These two outliers were taken from project feedback, and after looking at the .pdf they do seem
# rather obvious
data_dict.pop("THE TRAVEL AGENCY IN THE PARK") #Clearly not a person
data_dict.pop("LOCKHART EUGENE E") #Has only NaN values

print "I have removed the outliers"

print "There are ", len([data_dict[x] for x in data_dict if data_dict[x]["poi"] == 1]), "POIs and", len([data_dict[x] for x in data_dict if data_dict[x]["poi"] == 0]), "non-POIs"

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
for person in my_dataset:
    if my_dataset[person]["salary"] != "NaN" and my_dataset[person]["total_payments"] != "NaN":
        my_dataset[person]["salary_proportion_of_total_payments"] = my_dataset[person]["salary"] / float(my_dataset[person]["total_payments"])
    else:
        my_dataset[person]["salary_proportion_of_total_payments"] = "NaN"
    
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

# For the pipeline I am trying SVC here, and a decision tree/Naive Bayes in the commented out estimators
# I have included a Min Max Scaler and a PCA
SC = MinMaxScaler()
PC = PCA(svd_solver = "randomized", n_components = 7)
CG = GaussianNB()

estimators = [("Scale", SC), ("Reduce_Dimension", PC), ("Classifier", CG)]

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


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)