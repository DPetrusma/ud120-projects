#!/usr/bin/python

import sys
import pickle
from pprint import pprint
sys.path.append("../tools/")

### This is the same list of features given in the main project file
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

### Task 3: Loop through each feature and calculate the ratio of people with "NaN" for that feature to
### see if there are any features I should remove due to lack of data
features_list_empty_ratio = {}

for feature in features_list:
    features_list_empty_ratio[feature] = len([data_dict[i][feature] for i in data_dict if data_dict[i][feature] == "NaN"]) / float(len([data_dict[i][feature] for i in data_dict]))
    
pprint(features_list_empty_ratio)