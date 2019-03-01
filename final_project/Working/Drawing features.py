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
                'shared_receipt_with_poi'
                ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# The outlier I need to remove is the total - clearly not a person. I will check later for more
data_dict.pop("TOTAL")
data_dict.pop("MARTIN AMANDA K")

print "I have removed the outlier"

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

from draw_features import Draw2D, Draw3D

features_list_three = ['poi',
                # Financial features
                #'salary',
                ##'deferral_payments',
                #'total_payments',
                ##'loan_advances',
                'bonus',
                ##'restricted_stock_deferred',
                #'deferred_income',
                #'total_stock_value',
                #'expenses',
                #'exercised_stock_options',
                #'other',
                'long_term_incentive',
                'restricted_stock',
                ##'director_fees',
                # Email features
                #'to_messages',
                #'from_poi_to_this_person',
                #'from_messages',
                #'from_this_person_to_poi',
                #'shared_receipt_with_poi'
                ]

data_three = featureFormat(my_dataset, features_list_three, sort_keys = True)
labels, features_three = targetFeatureSplit(data_three)

Draw3D(features_three, labels, True, "image3D.png", features_list_three[1], features_list_three[2], features_list_three[3])