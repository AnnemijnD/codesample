import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from numpy.random import seed
import tensorflow as tf
import seaborn as sns
import time
from preprocessing import prep_data
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import pickle

sns.set()
sns.set_color_codes("pastel")

# def ncdg
def create_model(X_train, lyrs=[16], act="relu", opt='Adam', dr=0.2):
    """
    Creates neural network model with specified amount of layers and activation types.
    """

    # set random seed for reproducibility
    seed(42)
    tf.random.set_seed(42)

    model = Sequential()

    # create first hidden layer
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))

    # create additional hidden layers
    for i in range(1, len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))

    # dropout
    model.add(Dropout(dr))

    # create output layer
    model.add(Dense(3, activation="softmax"))  # output layer
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def create_prediction(df_test, X_train, y_train, X_test):
    """
    Create a prediction for the survival values of the testing set.
    """

    # make model: with or without dropout between hidden layers

    model = create_model(X_train)
    # model = create_dropout_model()
    # print(model.summary())

    # train model
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    y_train_hot = np_utils.to_categorical(encoded_Y)
    # for column in X_test.columns:
    #     if column not in X_train.columns:
    # print(X_test.columns)

    training = model.fit(X_train, y_train_hot, epochs=25, batch_size=50,\
                        validation_split=0.2, verbose=0)
    val_acc = np.mean(training.history['val_accuracy'])
    print("NN model validation accuracy during training: ", val_acc)


    pickle.dump(model, open("data/model2.pkl", "wb"))

    exit()
    # calculate predictions for test dataset
    # exit()


    class_predictions = model.predict_classes(X_test)
    predict_list = model.predict(X_test)

    category_p = []
    category_p1 = []
    category_p2 = []
    category_p3 = []
    for i in range(len(class_predictions)):
        category_p.append(predict_list[i][class_predictions[i]])
        category_p1.append(predict_list[i][0])
        category_p2.append(predict_list[i][1])
        category_p3.append(predict_list[i][2])

    df_test["category"] = class_predictions
    df_test["category_p"] = category_p
    # df_test["category_p1"] = category_p1
    # df_test["category_p2"] = category_p2
    # df_test["category_p3"] = category_p3
    # df['rank'] = df.groupby('srch_id')['category'].rank(ascending=True)
    # exit()
    solution = df_test[['srch_id', 'prop_id', 'category', "category_p"]]

    date_time = time.strftime("%Y-%m-%d-%H-%M")
    # solution.to_csv("results/unsorted" + str(date_time) + ".csv", index=False)

    # save prediction in output file

    solution = solution.sort_values(by=['srch_id', 'category', "category_p"], ascending=[True, False, False])
    solution = solution.drop("category", axis=1)
    solution = solution.drop("category_p", axis=1)

    # exit()
    solution.to_csv(f"data/results/NN_{date_time}.csv", index=False)

    return val_acc


# def create_model_testing(lyrs=[32], act="relu", opt='Adam', dr=0.2):
def create_model_testing(lyrs, act, opt='Adam', dr=0.2):
    """
    Creates neural network model with specified amount of layers and activation types.
    """

    # set random seed for reproducibility
    seed(42)
    tf.random.set_seed(42)

    # create sequential model
    model = Sequential()

    # create first hidden layer
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))

    # create additional hidden layers
    for i in range(1,len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))

    # add dropout, default is none
    model.add(Dropout(dr))

    # create output layer
    model.add(Dense(3, activation="softmax"))  # output layer
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def model_testing(X_train,y_train):
    """
    Run models with various activation methods and amounts of layers.
    """

    # for testing amount of layers, each layer has 32 neurons
    # layers = [[32, 32], [32, 32, 32], [32, 32, 32, 32], [32, 32, 32, 32],\
    #         [32, 32, 32, 32, 32], [32, 32, 32, 32, 32, 32]]
    layers = [[8], [16], [32], [64], [128], [256]]

    # activation = ["linear", "sigmoid", "relu", "softmax"]
    activation = ["relu"]
    runs = 1
    for i, act in enumerate(activation):
        val_accs = []
        for layer in layers:
            acc_avg = []
            for run in range(runs):
                model = create_model_testing(layer, act)

                # train model on full train set, with 80/20 CV split
                training = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)
                val_acc = np.mean(training.history['val_accuracy'])
                print("Run ", run, " - ", act + " activation - layer " + str(layer))
                acc_avg.append(val_acc)

            # save average accuracy of runs
            val_accs.append(round(np.mean(acc_avg)*100, 2))
            print("accuracy: " + str(np.mean(acc_avg)))

        # plot line for each activation method
        plt.plot([1,2,4,8,16,32,64,128,256], val_accs, label=act)
        # plt.plot(val_accs, label=act)

    # plotting
    plt.title("Accuracy of neural network model with different layers (N=" +\
            str(len(layers)) + ")", fontsize=22)
    plt.xlabel("Layers", fontsize=20)
    # plt.xticks(np.arange(1, len(val_accs) + 1, 1), fontsize=18)
    plt.ylabel("Accuracy (%)", fontsize=20)
    plt.legend()
    plt.subplots_adjust(bottom=.15, left=.15)
    plt.savefig("results/linear-relu-" + str(runs) + "runs.png")
    plt.show()


def param_testing(X_train, y_train):
    """
    Hyperparameter tuning.
    """

    model = KerasClassifier(build_fn=create_model, verbose=0)

    # define the grid search parameters
    batch_size = [10, 50, 100]
    epochs = [25, 50, 100]
    dr = [0.0, 0.2, 0.4]
    param_grid = dict(batch_size=batch_size, epochs=epochs, dr=dr)

    # search the grid
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        cv=5,
                        verbose=0)

    result = grid.fit(X_train, y_train)

    means = result.cv_results_['mean_test_score']
    stds = result.cv_results_['std_test_score']
    params = result.cv_results_['params']

    print(means)
    print(stds)
    print(params)

    print(grid.best_params_)
    print(f'Accuracy: {round(grid.best_score_*100, 2)}%')


if __name__ == "__main__":

    # preprocess data
    # df_train = pd.read_csv("/Volumes/Annemijn Dijkhuis/2nd-assignment-dmt-2020-2/training_set_VU_DM.csv")
    #
    # df_train = prep_data(df_train, "training")
    # df_train.to_csv("/Volumes/Annemijn Dijkhuis/2nd-assignment-dmt-2020-2/prep_data/training_set_prep.csv")
    # df_test = pd.read_csv("data/fake_data/test_fake.csv")
    # df_test = prep_data(df_test, "testing")

    df_test = pd.read_csv("data/fake_data/test_prep.csv")
    df_train = pd.read_csv("data/fake_data/training_prep.csv")

    # predicting columns of training set
    predictors = [c for c in df_train.columns if c not in ["prop_id","srch_id","booking_bool",\
                                "click_bool","gross_bookings_usd","position", "category", "visitor_hist_starrating", "visitor_hist_adr_usd"]]
    X_train = df_train[predictors]
    # X_train.drop(["srch_id", "prop_id"], axis=1, inplace=True)

    # predicting columns of test set
    cols = [col for col in df_test.columns if col not in ['prop_id', 'srch_id', "visitor_hist_starrating", "visitor_hist_adr_usd"]]
    X_test = df_test[cols]
    # X_test.drop(["srch_id", "prop_id"], axis=1, inplace=True)

    # prediction (outcome) variable
    # y_train = df_train.caategory.astype(int)
    # print(y_train.to_string())
    # exit()
    y_train = df_train["category"]
    # print(y_train)

    """ functions """
    create_model(X_train)
    # param_testing(X_train, y_train)
    # model_testing(X_train, y_train)
    # create_prediction(df_test, X_train, y_train, X_test)
