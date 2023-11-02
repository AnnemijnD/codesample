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

def process_data(argv):

    """
    Processes patient data to a usable format.

    Returns:
        X (numpy array) : numpy dataframe with chromosomal data
        Y (numpy array) : numpy dataframe with diagnosis per patient
    """

    #Storing the data as two Dataframes
    callfile = "data/Train_call.txt"
    clinicalfile = "data/Train_clinical.txt"

    dfcall = pd.read_csv(callfile, delimiter="\t")
    dfclin = pd.read_csv(clinicalfile, delimiter="\t")

    #Check whether there is any "null" or "na" in the table (there is not) so no need to print it
    dfcall.isnull().sum()
    dfcall.isna().sum()

    #Rotate dfcall 90 degrees. This is needed because we want to visualize information per patient.
    #To do so, data must be rotated so each row is now a patient with a specific combination of 0,1,2 values per column
    # (chromosome) and a diagnosis as last column

    temp_df = dfcall.T                      #Transposes the dataframe
    rotated_df=temp_df[4::]                 #Removes the first 4 lines corresponding to the chromosomal locations and clone number (we do not need them for the moment)
    rotated_df = rotated_df.reset_index()   #Sets the new index based on the new number of rows (needed for adding the Diagnosis column afterwards)
    #print("rot", rotated_df)

    #Add the column of the diagnosis from dfclin at the end of the rotated dfcall. Now each patient (row) has a combination
    # of 0,1,2 values, that we have to link to a diagnosis. It can be found in dfclin dataframe.

    final_df=rotated_df.assign(Diagnosis=dfclin.Subgroup)   #Adds a column Diagnosis with the information of dfclin "Subgroup" column

    # Store separately the values and the diagnosis. This step is needed for the classifier,
    # we need to give separately the values from the diagnosis

    X = final_df.iloc[:,1:2835].values      #Store in X all the row data (without sample name or diagnosis). NOTICE that this takes ALL the features, usually we would apply a feature selection method
    Y = final_df.iloc[:, -1].values         #Store in Y all the diagnosis ("Tripneg","HR+",...)

    return X, Y

def FS_ReliefF(X_train,Y_train,X_test,Nfeatures,ReliefF_K):
    """
    Feature selection using ReliefF

    Args:
        X (numpy array): aCGH data
        Y (numpy array): diagnosis data

    Returns:

        X_fil: filtered dataframe
    """
    fs = ReliefF(n_neighbors=ReliefF_K, n_features_to_keep=Nfeatures)
    fs.fit(X_train,Y_train)

    X_train_fil = fs.transform(X_train)
    X_test_fil = fs.transform(X_test)

    return X_train_fil,X_test_fil

def FS_RFE(X_train,Y_train,X_test,Nfeatures,RFE_step,classifier):
    """
    Feature selection using RFE-SVM

    Args:
        X (numpy array): aCGH data
        Y (numpy array): diagnosis data

    Returns:

        X_fil (numpy array): filtered dataframe
    """

    selector = RFE(classifier,n_features_to_select=Nfeatures,step=RFE_step)
    selector = selector.fit(X_train, Y_train)

    # construct mask
    mask = []
    for i in range(len(selector.support_)):
        if not selector.support_[i]:
            mask.append(i)

    X_train_fil = np.delete(X_train, mask, 1)
    X_test_fil = np.delete(X_test, mask, 1)

    return X_train_fil,X_test_fil

def FS_IG(X_train,Y_train,X_test,Nfeatures,IG_neighbours):
    """
    Feature selection using FS_IG

    Args:
        X (numpy array): aCGH data
        Y (numpy array): diagnosis data

    Returns:
        X_fil: filtered dataframe
    """

    # gets the gains vector
    gain_vec = mutual_info_classif(X_train, Y_train, discrete_features=True,n_neighbors=IG_neighbours)

    # gets the indices of columns that can be deleted from the dataset
    delete_ind = gain_vec.argsort()[::-1][Nfeatures:]

    # deletes the features that can be deleted
    X_train_fil = np.delete(X_train, delete_ind, 1)
    X_test_fil = np.delete(X_test, delete_ind, 1)

    return X_train_fil,X_test_fil

def classify(X_train, X_test, Y_train, Y_test,classifier):
    """

    Apply Support Vector Machine Algorithm for classification using the train set.

    """

    classifier.fit(X_train, Y_train)                        #Using the training data (X for the 0,1,2 and Y for the diagnosis associated)
    #Predict the test set results using SVM model
    Y_pred = classifier.predict(X_test)                     #This predicts the diagnosis (Y_pred) of the test set data (X_test)
    score = accuracy_score(Y_test,Y_pred)

    return score,Y_pred

def check_accuracy(highest_acc_params,X_train_out,Y_train_out,X_test_out,Y_test_out):

    """

    Calculates the accuracy of a model - derived from the parameter optimization in the
    inner folds of an outer fold - on the validation set of that outer fold.

    """

    c = highest_acc_params['inner_results']['c']
    max_iter = highest_acc_params['inner_results']['max_iter']
    selector = highest_acc_params['inner_results']['selector']
    Nfeatures = highest_acc_params['inner_results']['Nfeatures']

    classifier = SVC(kernel='linear',C=c,max_iter=max_iter)

    if selector == 'ReliefF':

        ReliefF_K = highest_acc_params['inner_results']['ReliefF_K']
        X_train_out_fil,X_test_out_fil = FS_ReliefF(X_train=X_train_out,Y_train=Y_train_out,X_test=X_test_out,Nfeatures=Nfeatures,ReliefF_K=ReliefF_K)
        score,Y_pred = classify(X_train_out_fil,X_test_out_fil,Y_train_out,Y_test_out,classifier)

    if selector == 'RFE':

        RFE_step = highest_acc_params['inner_results']['RFE_step']
        X_train_out_fil,X_test_out_fil = FS_RFE(X_train=X_train_out,Y_train=Y_train_out,X_test=X_test_out,Nfeatures=Nfeatures,RFE_step=RFE_step,classifier=classifier)
        score,Y_pred = classify(X_train_out_fil,X_test_out_fil,Y_train_out,Y_test_out,classifier)

    if selector == 'InfoGain':

        IG_neighbours = highest_acc_params['inner_results']['IG_neighbours']
        X_train_out_fil,X_test_out_fil = FS_IG(X_train=X_train_out,Y_train=Y_train_out,X_test=X_test_out,Nfeatures=Nfeatures,IG_neighbours=IG_neighbours)
        score,Y_pred = classify(X_train_out_fil,X_test_out_fil,Y_train_out,Y_test_out,classifier)

    outer_accuracy_results = {'score':score,'params':highest_acc_params,'Y_test':Y_test_out,'Y_pred':Y_pred}

    return outer_accuracy_results

def average_accuracy(inner_results,Nsplits_in):

    """

    Calculates the average accuracy of each set of parameters checked
    in the parameter optimization of each inner fold in an outer fold.

    """

    avg_accs = []
    highest_acc = 0
    highest_acc_params = {}

    for i in range(len(inner_results[1])):

        avg_acc = 0

        for ind_in,inner_result in inner_results.items():

            avg_acc += inner_results[ind_in][i]['score']

        avg_acc = avg_acc/Nsplits_in
        avg_accs.append({'avg_acc':avg_acc,'inner_result':inner_results[1][i]})

        if avg_acc > highest_acc:
            highest_acc = avg_acc
            highest_acc_params = inner_results[1][i]

    highest_acc_params = {'avg_acc':avg_acc,'inner_results':highest_acc_params}

    return highest_acc_params

def parameter_optimization(X_train_in,Y_train_in,X_test_in,Y_test_in,ind_in,inner_results):

    """

    Performs hyperparameter optimization for each inner fold.

    Adds to the 'inner_results' dictionary of the corresponding outer fold.

    """

    features = [10,20,30,40,50,60,70,80,90,100] # number of features
    # feature_selectors = ["ReliefF", "InfoGain", "RFE"]

    # # TODO:
    feature_selectors = ["RFE"]
    # classifier parameters
    cs = [1]
    max_iter_list = [900]

    # feature selector optimization
    RELIEFF_K_list = [7,8,9]
    RFE_step_list = [1,2]
    IG_neighbours_list = [1,2,3]

    for selector in feature_selectors:

        print('(checking {})'.format(selector))

        for Nfeatures in features:
            for c in cs:
                for max_iter in max_iter_list:

                    classifier = SVC(kernel='linear',C=c,max_iter=max_iter)

                    if selector == 'ReliefF':
                        for ReliefF_K in RELIEFF_K_list:

                            X_train_fil,X_test_fil = FS_ReliefF(X_train=X_train_in,Y_train=Y_train_in,X_test=X_test_in,Nfeatures=Nfeatures,ReliefF_K=ReliefF_K)
                            score,Y_pred = classify(X_train_fil,X_test_fil,Y_train_in,Y_test_in,classifier)
                            inner_results[ind_in].append({'selector':selector,'Nfeatures':Nfeatures,'score':score, 'c':c, 'max_iter':max_iter, 'ReliefF_K':ReliefF_K})

                    if selector == 'RFE':
                        for RFE_step in RFE_step_list:

                            X_train_fil,X_test_fil = FS_RFE(X_train=X_train_in,Y_train=Y_train_in,X_test=X_test_in,Nfeatures=Nfeatures,RFE_step=RFE_step,classifier=classifier)
                            score,Y_pred = classify(X_train_fil,X_test_fil,Y_train_in,Y_test_in,classifier)
                            inner_results[ind_in].append({'selector':selector,'Nfeatures':Nfeatures,'score':score, 'c':c, 'max_iter':max_iter, 'RFE_step':RFE_step})

                    if selector == 'InfoGain':
                        for IG_neighbours in IG_neighbours_list:

                            X_train_fil,X_test_fil = FS_IG(X_train=X_train_in,Y_train=Y_train_in,X_test=X_test_in,Nfeatures=Nfeatures,IG_neighbours=IG_neighbours)
                            score,Y_pred = classify(X_train_fil,X_test_fil,Y_train_in,Y_test_in,classifier)
                            inner_results[ind_in].append({'selector':selector,'Nfeatures':Nfeatures,'score':score, 'c':c, 'max_iter':max_iter, 'IG_neighbours':IG_neighbours})

def nested_cross_validate(X,Y,Nsplits_out,Nsplits_in,results,iter):

    """

    Performs a nested cross validation.

    Adds to the 'results' dictionary.

    """

    results[iter] = {}

    validator_out = KFold(n_splits=Nsplits_out,shuffle=True)

    ind_out = 0

    for train_index_out,test_index_out in validator_out.split(X):

        ind_out += 1

        print('\n starting with outer CV {} for iteration {}...\n'.format(ind_out,iter+1))

        X_train_out, X_test_out = X[train_index_out], X[test_index_out]
        Y_train_out, Y_test_out = Y[train_index_out], Y[test_index_out]

        validator_in = KFold(n_splits=Nsplits_in,shuffle=True)

        ind_in = 0

        inner_results = {}

        for train_index_in,test_index_in in validator_in.split(X_train_out):

            ind_in += 1

            inner_results[ind_in] = []

            print('starting with inner iteration {} of outer CV {} for iteration {}'.format(ind_in,ind_out,iter+1))

            X_train_in, X_test_in = X_train_out[train_index_in], X_train_out[test_index_in]
            Y_train_in, Y_test_in = Y_train_out[train_index_in], Y_train_out[test_index_in]

            parameter_optimization(X_train_in,Y_train_in,X_test_in,Y_test_in,ind_in,inner_results) #adds to inner_results

        highest_acc_params = average_accuracy(inner_results,Nsplits_in)
        outer_accuracy_results = check_accuracy(highest_acc_params,X_train_out,Y_train_out,X_test_out,Y_test_out)

        results[iter][ind_out] = outer_accuracy_results

if __name__ == "__main__":

    Nsplits_out = 3
    Nsplits_in = 5
    Niterations = 1

    results = {}

    X, Y = process_data(sys.argv)

    for iter in range(Niterations):

        nested_cross_validate(X,Y,Nsplits_out,Nsplits_in,results,iter) # adds to results

    # with open('results1.pkl', 'wb') as f:
    #     pickle.dump(results, f)

"""Sources:
https://towardsdatascience.com/building-a-simple-machine-learning-model-on-breast-cancer-data-eca4b3b99fa3
https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py"""
