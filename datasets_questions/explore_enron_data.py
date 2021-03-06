#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pprint

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print "Number of people: ", len(enron_data)
pprint.pprint(enron_data['TAYLOR MITCHELL S'])
print "Features per person: ", len(enron_data['TAYLOR MITCHELL S'])

print "Number of POI: ", sum( 1 for i in enron_data if enron_data[i]['poi'] == True)


