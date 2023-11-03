# Code Sample Bioinformatics project

This GitHub page contains the project I did during a bioinformatics course. We used machine learning to classify subtypes of breast cancer within patients.

## Overview

Here is an overview of the contents of this project. I would recommend to look at the nested_cross_validation.py file.
```
codesample
│   ├──  data                        Contains patient data
|   ├──  figures                     Contains figures shown in this README file
│   ├──  results                     
│   │       ├── features             Contains feature selection results
│   ├──  html                        Contains html files for figures
│   ├──  js                          Contains JavaScript files to create figures
│   │       ├── heatmap_RFE.js       JavaScript file that creates interactive heatmap
│   │       ├── scatterplot.js       JavaScript file that creates interactive scatterplot                 
│   ├── index.html                   Contains direct links to html files
│   ├── nested_cross_validation.py   Python file with nested cross validation
│   ├── train_final_model.py         Python file where the final model is trained


```


## Brief summary of the project
During this project, we obtained the data of breast cancer patients and were asked
to create a model that would be able to classify the samples into different breast cancer subtypes.
This project investigated three feature selection methods and used a Support
Vector Machine as a classification method. Using nested cross validation, the accuracy of the feature selection methods and several parameter settings was tested.

<html>
<p align="center">
<img src="figures/CV.png" width="50%" height="50%" class="center"/>
</p>
</html>

Ultimately, the best feature selection method and corresponding parameter set were chosen based on the accuracy scores that were found during cross validation.
We classified the breast cancer patients and created a confusion matrix of the results. The predicted class is shown on the X-axis
and on the Y-axis the true class can be observed.

<html>
<p align="center">
  <img src="figures/RFE_CM.png" width="50%" height="50%" class="center" />
</p>
</html>

Lastly, to manually investigate which features were selected most often,
an interactive scatterplot and frequency heat map was created with JavaScript. The scatterplot can be viewed [here](https://annemijnd.github.io/codesample/html/scat_freq_acc.html).  The heatmap can be viewed [here](https://annemijnd.github.io/codesample/html/heatmap_RFE.html).
