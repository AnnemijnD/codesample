import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import pickle
from nested_cross_validation import process_data

global CHOSEN_FEAUTURES
CHOSEN_FEAUTURES = []
N_FEATURES = 0

def FS_RFE(X, Y, step, c, max_iter):
    """
    Uses SVM-RFE feature selection to select features
    Args:
        X (NumPy array) : Array of patient data
        Y (NumPy array) : Array with subtype patient data
        step (int)      : RFE steps parameter
        c (int)         : SVM parameter
        max_iter (int)  : SVM parameter

    Returns:
        X_fil (NumPy array) : Array of patient data with only the selected features
    """

    # Define the SVM
    estimator = SVC(kernel = 'linear', max_iter=max_iter, C=c)

    # Define the feature selection method and select features
    selector = RFE(estimator,n_features_to_select=N_FEATURES, step=step)
    selector = selector.fit(X, y)

    # Construct mask
    mask = []

    for i in range(len(selector.support_)):
        if not selector.support_[i]:
            mask.append(i)
        else:

            # Save the features that were chosen
            CHOSEN_FEAUTURES.append(i)

    # Keep only selected features
    X_filtered = np.delete(X, mask, 1)

    return X_filtered

if __name__ == "__main__":
    save_data = False
    N_FEATURES = 20
    max_iter = 900
    c = 1
    step = 2

    # Load and preprocess the data
    X, y = process_data()

    # Perform feature selection
    X = FS_RFE(X, y, c, step, max_iter)

    # Build classifier
    classifier =  SVC(kernel = 'linear', max_iter=max_iter, C=c)
    classifier.fit(X, y)

    # save classifier
    if save_data:
        pickle.dump( [classifier, CHOSEN_FEAUTURES], open( "model.pkl", "wb" ) )
