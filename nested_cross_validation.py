import pandas as pd
import numpy as np
from ReliefF import ReliefF
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import itertools
import pickle
import sys

def process_data():
    """
    Processes patient data to a usable format.

    Returns:
        X (NumPy array) : NumPy array with chromosomal data
        y (NumPy array) : NumPy dataframe with diagnosis per patient
    """

    # Store the data in two Dataframes
    callfile = "data/Train_call.txt"
    clinicalfile = "data/Train_clinical.txt"

    dfcall = pd.read_csv(callfile, delimiter="\t")
    dfclin = pd.read_csv(clinicalfile, delimiter="\t")

    # Rotate dfcall 90 degrees, remove unnecessary lines and reset index
    temp_df = dfcall.T
    rotated_df=temp_df[4::]
    rotated_df = rotated_df.reset_index()

    # Add subgroup column to df
    final_df=rotated_df.assign(Diagnosis=dfclin.Subgroup)

    # Create patient data NumPy array
    X = final_df.iloc[:,1:len(dfcall) + 1].values

    # Create patient subtype data NumPy array
    y = final_df.iloc[:, -1].values

    return X, y

def FS_ReliefF(X_train,y_train,X_test,Nfeatures,ReliefF_K):
    """
    Uses ReliefF feature selection to select features

    Args:
        X_train (NumPy array) : Array of patient data of training set
        y_train (NumPy array) : Array with subtype patient data of training set
        X_test (NumPy array)  : Array of patient data of test set
        Nfeatures (int)       : Number of features to be selected
        RefiefF_K (int)       : ReliefF_K parameter

    Returns:
        X_train_fil (NumPy array) : Array of patient data with only the selected features (training set)
        X_test_fil (NumPy array)  : Array of patient data with only the selected features (test set)
    """

    # Select features
    fs = ReliefF(n_neighbors=ReliefF_K, n_features_to_keep=Nfeatures)
    fs.fit(X_train,y_train)

    # Keep only selected features
    X_train_fil = fs.transform(X_train)
    X_test_fil = fs.transform(X_test)

    return X_train_fil,X_test_fil

def FS_RFE(X_train,y_train,X_test,Nfeatures,RFE_step,classifier):
    """
    Uses RFE feature selection to select features

    Args:
        X_train (NumPy array)       : Array of patient data of training set
        y_train (NumPy array)       : Array with subtype patient data of training set
        X_test (NumPy array)        : Array of patient data of test set
        Nfeatures (int)             : Number of features to be selected
        RFE_step (int)              : RFE steps parameter
        classifier (sklearn object) : The SVM classifier object

    Returns:
        X_train_fil (NumPy array) : Array of patient data with only the selected features (training set)
        X_test_fil (NumPy array)  : Array of patient data with only the selected features (test set)
    """

    # Select features
    selector = RFE(classifier,n_features_to_select=Nfeatures,step=RFE_step)
    selector = selector.fit(X_train, y_train)

    # Construct mask
    mask = []
    for i in range(len(selector.support_)):
        if not selector.support_[i]:
            mask.append(i)

    # Keep only selected features
    X_train_fil = np.delete(X_train, mask, 1)
    X_test_fil = np.delete(X_test, mask, 1)

    return X_train_fil,X_test_fil

def FS_IG(X_train,y_train,X_test,Nfeatures,IG_neighbours):
    """
    Uses InfoGain feature selection to select features

    Args:
        X_train (NumPy array) : Array of patient data of training set
        y_train (NumPy array) : Array with subtype patient data of training set
        X_test (NumPy array)  : Array of patient data of test set
        Nfeatures (int)       : Number of features to be selected
        IG_neighbours (int)   : InfoGain parameter

    Returns:
        X_train_fil (NumPy array) : Array of patient data with only the selected features (training set)
        X_test_fil (NumPy array)  : Array of patient data with only the selected features (test set)
    """

    # Get the gains vector
    gain_vec = mutual_info_classif(X_train, y_train, discrete_features=True,n_neighbors=IG_neighbours)

    # Obtain indices of columns to be deleted
    delete_ind = gain_vec.argsort()[::-1][Nfeatures:]

    # Keep only selected features
    X_train_fil = np.delete(X_train, delete_ind, 1)
    X_test_fil = np.delete(X_test, delete_ind, 1)

    return X_train_fil,X_test_fil

def classify(X_train, X_test, y_train, y_test, classifier):
    """
    Uses Support Vector Machine to classify patient data.

    Args:
        X_train (NumPy array)       : Array of patient data of training set
        X_test (NumPy array)        : Array of patient data of test set
        y_train (NumPy array)       : Array with subtype patient data of training set
        y_test (NumPy array)        : Array of subtype patient data of test set
        classifier (sklearn object) : The SVM classifier object

    Returns:
        score (float)           : Accuracy score of the fitting
        y_pred (NumPy array)    : Classified patient samples
    """

    # Fit the data on the training set
    classifier.fit(X_train, y_train)

    # Predict the test set results using SVM model
    y_pred = classifier.predict(X_test)

    # Obtain accuracy
    score = accuracy_score(y_test,y_pred)

    return score,y_pred

def check_accuracy(highest_acc_params,X_train_out,y_train_out,X_test_out,y_test_out):
    """
    Calculates the accuracy of a model - derived from the parameter optimization in the
    inner folds of an outer fold - on the test set of that outer fold.

    Args:
        highest_acc_params (dict)   : Best parameters found during parameter optimization all inner folds of one outer fold
        X_train_out (NumPy array)   : Outer fold patient data of the training set
        y_train_out (NumPy array)   : Outer fold patient subtype data of the training set
        X_test_out (NumPy array)    : Outer fold patient data of the test set
        y_test_out (NumPy array)    : Outer fold patient subtype data of the test set

    Returns:
        outer_accuracy_results (dict) : accuracy of the model with selected parameters
    """

    # Obtain the parameter set with the highest accuracy of the inner folds of one outer fold
    c = highest_acc_params['inner_results']['c']
    max_iter = highest_acc_params['inner_results']['max_iter']
    selector = highest_acc_params['inner_results']['selector']
    Nfeatures = highest_acc_params['inner_results']['Nfeatures']

    # Define SVM classifier with found parameters
    classifier = SVC(kernel='linear',C=c,max_iter=max_iter)

    # Perform feature selection based on which method was found best and fit data
    if selector == 'ReliefF':

        ReliefF_K = highest_acc_params['inner_results']['ReliefF_K']
        X_train_out_fil,X_test_out_fil = FS_ReliefF(X_train_out,y_train_out,X_test_out,Nfeatures,ReliefF_K)
        score,y_pred = classify(X_train_out_fil,X_test_out_fil,y_train_out,y_test_out,classifier)

    elif selector == 'RFE':

        RFE_step = highest_acc_params['inner_results']['RFE_step']
        X_train_out_fil,X_test_out_fil = FS_RFE(X_train_out,y_train_out,X_test_out,Nfeatures,RFE_step,classifier)
        score,y_pred = classify(X_train_out_fil,X_test_out_fil,y_train_out,y_test_out,classifier)

    elif selector == 'InfoGain':

        IG_neighbours = highest_acc_params['inner_results']['IG_neighbours']
        X_train_out_fil,X_test_out_fil = FS_IG(X_train_out,y_train_out, X_test_out,Nfeatures,IG_neighbours)
        score,y_pred = classify(X_train_out_fil,X_test_out_fil,y_train_out,y_test_out,classifier)

    # Save the accuracy score of this outer fold
    outer_accuracy_results = {'score':score,'params':highest_acc_params,'y_test':y_test_out,'y_pred':y_pred}

    return outer_accuracy_results

def average_accuracy(inner_results,Nsplits_in):
    """
    Calculates the average accuracy of each set of parameters checked
    in the parameter optimization of each inner fold in one outer fold.

    Args:
        inner_results (dict)    : A dictionary of all inner fold scores
        Nsplits_in (int)        : The number of inner fold splits

    Returns:
        highest_acc_params (dict) : The parameter set that had the highest average accuracy in all inner folds
    """

    # Create a list of all average accuracies
    avg_accs = []
    highest_acc = 0
    highest_acc_params = {}

    # Loop through the results of the parameter optimization
    for i in range(len(inner_results[1])):

        # Set standard accuracy to 0
        avg_acc = 0

        for ind_in,inner_result in inner_results.items():

            avg_acc += inner_results[ind_in][i]['score']

        # Get average accuracy of one set of inner folds
        avg_acc = avg_acc/Nsplits_in
        avg_accs.append({'avg_acc':avg_acc,'inner_result':inner_results[1][i]})

        # Any time higher accuracy is found, replace with new accuracy
        if avg_acc > highest_acc:
            highest_acc = avg_acc
            highest_acc_params = inner_results[1][i]

    # Create dictionary with the parameters with the best found average accuracy
    highest_acc_params = {'avg_acc':avg_acc,'inner_results':highest_acc_params}

    return highest_acc_params

def parameter_optimization(X_train_in,y_train_in,X_test_in,y_test_in,ind_in,inner_results):
    """
    Performs hyperparameter optimization for each inner fold and updates
    the 'inner_results' dictionary.

    Args:
        X_train_in (NumPy array)    : Inner fold patient data of the training set
        y_train_in (NumPy array)    : Inner fold patient subtype data of the training set
        X_test_in (NumPy array)     : Inner fold patient data of the test set
        y_test_in (NumPy array)     : Inner fold patient subtype data of the test set
        ind_in (int)                : Inner fold iteration were are currently at
        inner_results (dict)        : The dictionary with results of all previous inner folds

    Returns:
        inner_results (dict)        : Updated dictionary with results of this and all previous inner folds
    """

    # number of features
    features = [10,20,30,40,50,60,70,80,90,100]
    feature_selectors = ["ReliefF", "InfoGain", "RFE"]

    # classifier parameters
    cs = [1]
    max_iter_list = [900]

    # feature selector optimization
    RELIEFF_K_list = [7,8,9]
    RFE_step_list = [1,2]
    IG_neighbours_list = [1,2,3]

    # Loop different all parameter sets
    for selector in feature_selectors:

        print('(checking {})'.format(selector))

        for Nfeatures in features:
            for c in cs:
                for max_iter in max_iter_list:

                    # Define classifier
                    classifier = SVC(kernel='linear',C=c,max_iter=max_iter)

                    # Perform feature selection and classify the data with the SVM
                    if selector == 'ReliefF':
                        for ReliefF_K in RELIEFF_K_list:

                            X_train_fil,X_test_fil = FS_ReliefF(X_train_in,y_train_in,X_test_in, Nfeatures,ReliefF_K)
                            score,y_pred = classify(X_train_fil,X_test_fil,y_train_in,y_test_in,classifier)
                            inner_results[ind_in].append({'selector':selector,'Nfeatures':Nfeatures,'score':score, 'c':c, 'max_iter':max_iter, 'ReliefF_K':ReliefF_K})

                    elif selector == 'RFE':
                        for RFE_step in RFE_step_list:

                            X_train_fil,X_test_fil = FS_RFE(X_train_in,y_train_in,X_test_in,Nfeatures,RFE_step,classifier=classifier)
                            score,y_pred = classify(X_train_fil,X_test_fil,y_train_in,y_test_in,classifier)
                            inner_results[ind_in].append({'selector':selector,'Nfeatures':Nfeatures,'score':score, 'c':c, 'max_iter':max_iter, 'RFE_step':RFE_step})

                    elif selector == 'InfoGain':
                        for IG_neighbours in IG_neighbours_list:

                            X_train_fil,X_test_fil = FS_IG(X_train_in,y_train_in,X_test_in,Nfeatures,IG_neighbours)
                            score,y_pred = classify(X_train_fil,X_test_fil,y_train_in,y_test_in,classifier)
                            inner_results[ind_in].append({'selector':selector,'Nfeatures':Nfeatures,'score':score, 'c':c, 'max_iter':max_iter, 'IG_neighbours':IG_neighbours})

    return inner_results

def nested_cross_validate(X,y,Nsplits_out,Nsplits_in,results,iter):
    """
    Performs a nested cross validation and updates the 'results' dictionary.

    Args:
        X (NumPy array)     : All patient data
        y (NumPy array)     : All patient subtype data
        Nsplits_out (int)   : Amount of inner folds of cross validation
        Nsplits_in (int)    : Amount of outer folds of cross validation
        results (dict)      : Dictionary of all previous results of outer folds
        iter (int)          : The iteration of nested cross validation were are currently in

    Returns:
        results (dict)      : Updated dictionary of this and all previous results of outer folds
    """

    # Add dictionary to results dictionary
    results[iter] = {}

    # Define Kfold cross validation object
    validator_out = KFold(n_splits=Nsplits_out,shuffle=True)

    # Counter of iteration outer fold
    ind_out = 0

    # Loop through outer fold splits
    for train_index_out,test_index_out in validator_out.split(X):

        ind_out += 1

        print('\n starting with outer CV {} for iteration {}...\n'.format(ind_out,iter+1))

        # Create outer fold data sets
        X_train_out, X_test_out = X[train_index_out], X[test_index_out]
        y_train_out, y_test_out = y[train_index_out], y[test_index_out]

        # Define inner fold KFold object
        validator_in = KFold(n_splits=Nsplits_in,shuffle=True)

        # Counter of iteration inner fold
        ind_in = 0

        # Define inner fold results dictionary
        inner_results = {}

        # Loop through inner fold splits
        for train_index_in,test_index_in in validator_in.split(X_train_out):

            ind_in += 1

            inner_results[ind_in] = []

            print('starting with inner iteration {} of outer CV {} for iteration {}'.format(ind_in,ind_out,iter+1))

            # Create inner fold data set
            X_train_in, X_test_in = X_train_out[train_index_in], X_train_out[test_index_in]
            y_train_in, y_test_in = y_train_out[train_index_in], y_train_out[test_index_in]

            # Perform parameter optimization and update inner_results
            inner_results = parameter_optimization(X_train_in,y_train_in,X_test_in,y_test_in,ind_in,inner_results)

        # Define the average accuracy of all inner folds within this outer fold
        highest_acc_params = average_accuracy(inner_results,Nsplits_in)

        # Define the accuracy of this outer fold using outer fold test set
        outer_accuracy_results = check_accuracy(highest_acc_params,X_train_out,y_train_out,X_test_out,y_test_out)

        # Update results dictionary
        results[iter][ind_out] = outer_accuracy_results

    return results

if __name__ == "__main__":
    save_data = False

    # Number of splits in the cross validation in the outer and inner folds
    Nsplits_out = 3
    Nsplits_in = 5

    # Number of iterations
    Niterations = 10

    # Define results dictionary
    results = {}

    # Load and preprocess the data
    X, y = process_data()

    # Start nested cross validation
    for iter in range(Niterations):

        results = nested_cross_validate(X,y,Nsplits_out,Nsplits_in,results,iter)

    if save_data:
        with open('results.pkl', 'wb') as f:
            pickle.dump(results, f)

"""Sources:
https://towardsdatascience.com/building-a-simple-machine-learning-model-on-breast-cancer-data-eca4b3b99fa3
https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py"""
