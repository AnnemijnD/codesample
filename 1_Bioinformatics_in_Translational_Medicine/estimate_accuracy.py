import pickle
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

#get all seperate pkl files from the folder results
#each file contains 1 iterations of 3 outer folds of the nested cros validation
all_files = [f for f in listdir('results/runs') if isfile(join('results/runs', f))]

#create list to combine all seperate iteration
combined_results_NCV = []

for f in all_files:
    f = "results/runs/" + f
    with open(f, 'rb') as f1:
        f2 = pickle.load(f1)
        #put all seperate iteration pkl files, in one list
        combined_results_NCV.append(f2)


#remove "score" from inner_results. we don't need those and they make it impossible to make a distinction between parametes.
for iter in combined_results_NCV:
    for i in range (1,4):
        #add inner results to inner_results list
        del iter[0][i]["params"]["inner_results"]["score"]

#get average outer accuracy per parameter group
#so get outer accuracy for each distinct inner_results
distinct_inner_results = []

for iter in combined_results_NCV:
    for i in range (1,4):
        #add inner results to inner_results list
        inner_results = iter[0][i]["params"]["inner_results"]
        distinct_inner_results.append(str(inner_results))



#Make list distinct_inner_results really distinct
distinct_inner_results = list(set(distinct_inner_results))


#iterate over distinct inner results and get outer fold accuracy.
results = []

for inner_result in distinct_inner_results:

    outer_fold_score_sum = 0
    freq = 0
    for iter in combined_results_NCV:
        for i in range (1,4):

            # find the outer fold accuracy of the distinct models
            if str(iter[0][i]["params"]["inner_results"]) == inner_result:
                outer_fold_score_sum = outer_fold_score_sum + iter[0][i]["score"]
                freq += 1
    results.append({ "inner_result" : inner_result, "outer_fold_score_sum" : outer_fold_score_sum, "freq" : freq} )


#so results[] contains for each distict model: sum_outer_fold_score and freq.
#now we want to calculate the average accuracy (= sum /feq)
for result in results:
    result["average_accuracy"] = result["outer_fold_score_sum"] / result["freq"]

ordered_results = sorted(results, key = lambda k: k["average_accuracy"])

# put ordered_results list in a dataframe so it readable
df_results = pd.DataFrame(ordered_results)
print(df_results, "\n ")

# print only the models so it's readable.
u = 0
for result in ordered_results:
    print(u, "     ", result["inner_result"])
    u = u + 1

##################################   MAKING THE Confusion Matrix:
# Based on df_results...
# We decided that the best model is model number 10. Selector= RFE and N features = 20 RFE_step=2 so select that one for confusion matrix to estimate accuracies
best_model=[]
for iteration in combined_results_NCV:
    for i in range(1, 4):
        if (iteration[0][i]["params"]["inner_results"]["selector"]) == "RFE" and (iteration[0][i]["params"]["inner_results"]["Nfeatures"]==20) and (iteration[0][i]["params"]["inner_results"]["RFE_step"]==2):
            best_model.append(iteration[0][i])

#The model selected was used three times. Create a confusion matrix for each one of them and sum them to obtain the final confusion matrix
cm_final=0
labels = ["HER2+", "HR+", "Triple Neg"]
for model in best_model:
    cm=confusion_matrix(model["Y_test"], model["Y_pred"], labels=labels)
    cm_final=cm_final + cm

#Calculate accuracy of the confusion matrix obtained from the three times
ac_final=sum(np.diag(cm_final))/sum(sum(cm_final))
print("The total accuracy is" , ac_final)

#Calculate the accuracy for each class, for a proper estimation of the nº of misclassifications per class
ac_HER=cm_final[0][0]/(cm_final[0][0]+cm_final[1][0]+cm_final[2][0])
ac_HR=cm_final[1][1]/(cm_final[0][1]+cm_final[1][1]+cm_final[2][1])
ac_TN=cm_final[2][2]/(cm_final[0][2]+cm_final[1][2]+cm_final[2][2])
print("The accuracy for HER type is",ac_HER, "\n The accuracy for HR type is",ac_HR,"\n The accuracy for TN type is",ac_TN)

#Calculate the estimated of the amount of correctly predicted samples in the final prediction:
#so perdicted_HER, predicted_HR and predicted_TN, need to be adjusted to the numer of times these types are predicted on the prediction_data
predicted_HER=19
predicted_HR=24
predicted_TN=14

error_HER=(ac_HER)*predicted_HER
error_HR=(ac_HR)*predicted_HR
error_TN=(ac_TN)*predicted_TN

total_error=error_HER+error_HR+error_TN
print('Our estimate for the number of correctly predicted cases is: ', round(total_error))

# Visualize the confusion table
sns.set(font_scale=1.4)  # for label size
sns.heatmap(cm_final, annot=True, annot_kws={"size": 16}, xticklabels=labels, yticklabels=labels,
            cmap="Blues",cbar=False)
plt.title("Confusion matrix from the selected model")
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()








#
