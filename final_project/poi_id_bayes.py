def TestGaussianClassifier(features, labels, sss, scoring = "f1", features_list = None):

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_selection import SelectKBest
    from sklearn.naive_bayes import GaussianNB

    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    from pprint import pprint

    print "I am running a Naive Bayes Classifier"
    
    # For the pipeline I am trying Naive Bayes here, and I have included a Min Max Scaler and select k best
    estimators = [("Scale", MinMaxScaler()), ("Feature_selection", SelectKBest()), ("Classifier", GaussianNB())]

    # Create the pipeline
    pipe = Pipeline(estimators)

    # These are the parameters to loop through for the different classifiers
    params = {  "Feature_selection__k" : range(1, 16)
            }
      
    # Now, run the grid search to find the best combination of parameters to maximise the f1 score
    # Since I include the stratified shuffle split in the grid search, I don't need "features_test" and "features_train"
    #print "I have set up the Grid Search"

    clf_gs = GridSearchCV(pipe, params, cv = sss, scoring = scoring)
    clf_gs.fit(features, labels)

    #print "I have fit the grid search"

    # Print out the best parameters so that I can use the best one for clf
    print "Best parameters for Naive Bayes: ", pprint(clf_gs.best_params_)
    print "Feature scores from Select K Best for Naive Bayes: ", pprint(zip(features_list, clf_gs.best_estimator_.named_steps['Feature_selection'].scores_))
    
    print "Classification Report for Naive Bayes: "
    print classification_report(y_true = labels, y_pred = clf_gs.predict(features), target_names = ["Not POI", "POI"])

    return clf_gs