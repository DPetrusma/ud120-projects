#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    #print predictions[0][0]
    #print net_worths[0][0]
    #print net_worths[0][0] - predictions[0][0]
    
    ### your code goes here
    #Since the input lists are actually lists of 1d arrays, I want to take [i][0] to end up with
    # a list of tuples of floats in cleaned_data.
    for i in range(len(ages)):
        cleaned_data.append((ages[i][0], net_worths[i][0], net_worths[i][0] - predictions[i][0]))
        
    #I will sort cleaned_data by the third element (the error) squared, then take the first 90%
    cleaned_data = sorted(cleaned_data, key = lambda x: (x[2]**2))
    cleaned_data = cleaned_data[0:int(len(cleaned_data)*0.9)]

    print "Remaining elements in train data after cleaning: ", len(cleaned_data)
    
    return cleaned_data

