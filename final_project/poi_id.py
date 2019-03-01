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
                'from_this_person_to_poi',
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

def discoverBestClassifier(features, labels):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_selection import SelectKBest
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score

    #I will use these individual functions to test each of the classifiers I am interested in
    from poi_id_bayes import TestGaussianClassifier
    from poi_id_SVC import TestSVCClassifier
    from poi_id_Random_Forest import TestRandomForestClassifier

    print "I have imported modules"

    #I am running only 100 folds that this actually finishes
    sss = StratifiedShuffleSplit(n_splits = 100, test_size = 0.1, random_state = 42)
    
    
    clf_gs_Gauss = TestGaussianClassifier(features, labels, sss, "f1_weighted", features_list[1:])

    #https://stackoverflow.com/questions/33376078/python-feature-selection-in-pipeline-how-determine-feature-names
    bf = clf_gs_Gauss.best_estimator_.named_steps["Feature_selection"].get_support(indices = True)
    #This is +1 since we have POI in the features list, which is actually the label
    print "Best k features for Naive Bayes: ", [features_list[i + 1] for i in bf]
    
    
    clf_gs_SVC = TestSVCClassifier(features, labels, sss, "f1_weighted", features_list[1:])

    bf = clf_gs_SVC.best_estimator_.named_steps["Feature_selection"].get_support(indices = True)
    #This is +1 since we have POI in the features list, which is actually the label
    print "Best k features for SVC ", [features_list[i + 1] for i in bf]
    
    clf_gs_Random_Forest = TestRandomForestClassifier(features, labels, sss, "f1_weighted", features_list[1:])

    bf = clf_gs_Random_Forest.best_estimator_.named_steps["Feature_selection"].get_support(indices = True)
    #This is +1 since we have POI in the features list, which is actually the label
    print "Best k features for Random Forest: ", [features_list[i + 1] for i in bf]
    
    #Orignally I tried this, but the best f1 score here doesn't give the best tester score
    #from tester import test_classifier
    
    #This is where I figure out which one is best and choose it
    best_f1_score = ["None", 0, None]
    
    f = f1_score( y_true = labels, y_pred = clf_gs_Gauss.predict(features), average = "weighted")
    #print "Tester results: ", test_classifier(clf_gs_Gauss, my_dataset, features_list)

    if f >= best_f1_score[1]:
        best_f1_score = ["Gauss", f, clf_gs_Gauss]
    
    f = f1_score( y_true = labels, y_pred = clf_gs_SVC.predict(features), average = "weighted")
    #print "Tester results: ", test_classifier(clf_gs_SVC, my_dataset, features_list)
    
    if f >= best_f1_score[1]:
        best_f1_score = ["SVC", f, clf_gs_SVC]
           
    f = f1_score( y_true = labels, y_pred = clf_gs_Random_Forest.predict(features), average = "weighted")
    #print "Tester results: ", test_classifier(clf_gs_Random_Forest, my_dataset, features_list)

    if f >= best_f1_score[1]:
        best_f1_score = ["Random Forest", f, clf_gs_Random_Forest]
      
    print "Best f1 score was", best_f1_score[1], "coming from ", best_f1_score[0]
    
    all_best_classifiers = [best_f1_score, clf_gs_Gauss, clf_gs_SVC, clf_gs_Random_Forest]
    #all_best_classifiers = [best_f1_score, clf_gs_Gauss, None, None]
       
    return all_best_classifiers

all_best_classifiers = discoverBestClassifier(features, labels)
best_classifier = all_best_classifiers[0]

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#After running the above funciton, I know what the best classifier and parameters are to use

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
   
# For the pipeline I am using scaling, the estimator and the k features from the best classifier
SC = MinMaxScaler()
KC = best_classifier[2].best_estimator_.named_steps['Feature_selection']
#CG = GaussianNB()
CG = best_classifier[2].best_estimator_.named_steps['Classifier']

estimators = [("Scale", SC), ("Select_Features", KC), ("Classifier", CG)]

# Create the pipeline
clf = Pipeline(estimators)

from tester import test_classifier

test_classifier(clf, my_dataset, features_list)

#Now I want to try all versions

for c in range(1,4):
    KC = all_best_classifiers[c].best_estimator_.named_steps['Feature_selection']
    CG = all_best_classifiers[c].best_estimator_.named_steps['Classifier']
    estimators = [("Scale", SC), ("Select_Features", KC), ("Classifier", CG)]
    clf = Pipeline(estimators)

    test_classifier(clf, my_dataset, features_list)


#This is where I manually choose the best one from the tester results if they differ from the best f1 scores
clf = all_best_classifiers[1]

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)