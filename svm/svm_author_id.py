#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn import svm

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

#########################################################
t0 = time()

#C = 10: 0.616040955631
#C = 100: 0.616040955631
#C = 1000: 0.821387940842
#C = 10000: 0.892491467577

clf = svm.SVC(kernel = "rbf", C = 10000.0)
clf.fit(features_train, labels_train)

print "training time:", round(time()-t0, 3), "s"
t0 = time()

prediction_array = clf.predict(features_test)

print "predicting time:", round(time()-t0, 3), "s"
t0 = time()

print clf.score(features_test, labels_test)

print "measuring time:", round(time()-t0, 3), "s"

print prediction_array
print prediction_array[10]
print prediction_array[26]
print prediction_array[50]

print sum( 1 for i in prediction_array if i == 1 )
print sum( 1 for i in prediction_array if i == 0 )