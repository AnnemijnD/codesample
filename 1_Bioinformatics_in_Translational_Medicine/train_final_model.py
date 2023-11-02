import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import pickle

global CHOSEN_FEAUTURES
CHOSEN_FEAUTURES = []
N_FEATURES = 0

def process_data():
    """
    Processes patient data to a usable format.

    Returns:
        X (NumPy array) : numpy dataframe with chromosomal data
        Y (NumPy array) : numpy dataframe with diagnosis per patient
    """

    # Store the data in two Dataframes
    callfile = "data/Train_call.txt"
    clinicalfile = "data/Train_clinical.txt"

    dfcall = pd.read_csv(callfile, delimiter="\t")
    dfclin = pd.read_csv(clinicalfile, delimiter="\t")

    # Check whether there is any "null" or "na" in the table
    dfcall.isnull().sum()
    dfcall.isna().sum()

    # Rotate dfcall 90 degrees, remove unnecessary lines and reset index
    temp_df = dfcall.T
    rotated_df=temp_df[4::]
    rotated_df = rotated_df.reset_index()

    # Add subgroup column to df
    final_df=rotated_df.assign(Diagnosis=dfclin.Subgroup)

    # Create patient data NumPy array
    X = final_df.iloc[:,1:2835].values

    # Create patient subtype data NumPy array
    Y = final_df.iloc[:, -1].values

    return X, Y


def FS_RFE(X, Y, step, c, max_iter):
    """
    Uses SVM-RFE feature selection to select features
    Args:
        X (numpy array) : Array of patient data
        Y (numpy array) : Array with subtype patient data
        step (int)      : RFE steps parameter
        c (int)         : SVM parameter
        max_iter (int)  : SVM parameter

    Returns:
        X_fil (numpy array) : Array of patient data with only the selected features
    """

    # Define the SVM
    estimator = SVC(kernel = 'linear', max_iter=max_iter, C=c)

    # Define the feature selection method and select features
    selector = RFE(estimator,n_features_to_select=N_FEATURES, step=step)
    selector = selector.fit(X, Y)

    # Construct mask
    mask = []
    for i in range(len(selector.support_)):
        if not selector.support_[i]:
            mask.append(i)
        else:
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
    X, Y = process_data()

    # Perform feature selection
    X = FS_RFE(X=X, Y=Y, c=c, step=step, max_iter=max_iter)

    # Build classifier
    classifier =  SVC(kernel = 'linear', max_iter=max_iter, C=c)
    classifier.fit(X, Y)

    # save classifier
    if save_data:
        pickle.dump( [classifier, CHOSEN_FEAUTURES], open( "model.pkl", "wb" ) )
