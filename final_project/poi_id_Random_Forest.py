def TestRandomForestClassifier(features, labels, sss, scoring = "f1", features_list = None):

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_selection import SelectKBest
    from sklearn.ensemble import RandomForestClassifier

    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    from pprint import pprint
    
    print "I am running a random forest classifier"

    # For the pipeline I am trying a random forest here, and I have included a Min Max Scaler and select k best
    estimators = [("Scale", MinMaxScaler()), ("Feature_selection", SelectKBest()), ("Classifier", RandomForestClassifier())]

    # Create the pipeline
    pipe = Pipeline(estimators)

    # These are the parameters to loop through for the different classifiers
    params = {  "Feature_selection__k" : range(3, 16),
                "Classifier__min_samples_split" : range(2,20),
                "Classifier__min_samples_leaf" : range(1,5),
                "Classifier__criterion" : ["gini"] #, "entropy"]
            }
        
    # Now, run the grid search to find the best combination of parameters to maximise the f1 score
    # Since I include the stratified shuffle split in the grid search, I don't need "features_test" and "features_train"
    #print "I have set up the Grid Search"

    clf_gs = GridSearchCV(pipe, params, cv = sss, scoring = scoring)
    clf_gs.fit(features, labels)

    #print "I have fit the grid search"

    # Print out the best parameters so that I can use the best one for clf
    print "Best parameters for Random Forest: ", pprint(clf_gs.best_params_)
    print "Feature scores from Select K Best for Random Forest: ", pprint(zip(features_list, clf_gs.best_estimator_.named_steps['Feature_selection'].scores_))
    print "Feature importance for Random Forest: ", clf_gs.best_estimator_.named_steps['Classifier'].feature_importances_
    
    print "Classification Report for Random Forest: "
    print classification_report(y_true = labels, y_pred = clf_gs.predict(features), target_names = ["Not POI", "POI"])

    return clf_gs