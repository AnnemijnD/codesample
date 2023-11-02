# Code Sample

## Files
This folder contains the projects of two courses I followed during my master's.
1. Bioinformatics in Translational Medicine
2. Datamining


## 1. Bioinformatics in Translational Medicine

### Brief summary of the project
During this project, we obtained the data of 100 breast cancer patients and were asked
to create a model that would be able to classify the samples into different breast cancer subtypes.
This project investigated three feature selection methods and used a Support
Vector Machine as a classification method. Using nested cross validation, the accuracy of the feature selection methods and several parameter settings was tested.


![ScreenShot](Figures/CV.png)

Ultimately, the best feature selection method and corresponding parameter set were chosen based on the accuracy scores that were found during cross validation.
We classified the breast cancer patients and created a confusion matrix of the results. The predicted class is shown on the X-axis
and on the Y-axis the true class can be observed.

![ScreenShot](/Figures/RFE_CM.png)

Lastly, to manually investigate which features were selected most often,
an interactive frequency heat map was created with. This heatmap can be viewed [here](https://annemijnd.github.io/codesample/1_Bioinformatics_in_Translational_Medicine/html/heatmap_RFE.html). In the upper graph, the features are sorted
by frequency. In the bottom graph, they are ordered by feature number.

### Files and code
```
1_Bioinformatics_in_Translational_Medicine
│   ├── data
│   ├── seperate_iterations_results_NCV
│   ├── estimate_accuracy.py
│   ├──nesed_cross_validation.py
│   ├──train_final_model.py


```


## 2. Datamining
