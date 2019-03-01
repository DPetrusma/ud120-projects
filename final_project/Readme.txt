All three of the below files are imported and run by poi_id.py

poi_id_bayes.py: This file contains my grid search for a Naive Bayes classifier to find the optimal number of features to keep. If the best result was higher than the best result for the other classifier grid searches (which it was), I would use that in my final algorithm.
poi_id_Random_Forest.py: This file contains my grid search for a Random Forest classifier to find the optimal number of features to keep, as well as the optimal values for several other parameters. If the best result was higher than the best result for the other classifier grid searches, I would use that in my final algorithm.
poi_id_SVC.py: This file contains my grid search for a Support Vector Machine classifier to find the optimal number of features to keep, as well as the optimal values for several other parameters. If the best result was higher than the best result for the other classifier grid searches, I would use that in my final algorithm.

Sources: sklearn documentation (http://scikit-learn.org/stable/index.html) and Udacity Forum (https://discussions.udacity.com/c/nd002-intro-to-machine-learning)