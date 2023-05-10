
# EEG-Based Eye State Classification
`<br>
![Deep learning white](https://github.com/SimoneParvizi/ML-EEG-eyes-model/assets/75120707/db9f8903-d045-492d-9c0c-3d315b323dc7)

## Introduction
Approximately 30% of EEG diagnoses are misread and misdiagnosed due to human errors, which is likely to be a greater number when considering all diagnosable diseases. AI architectures are being used in various fields like infant sleep-waking state identification, detection of driver drowsiness, classification of bipolar mood disorder, and ADHD patient diagnosis.   
  
ML cross-validation methods such as train-test splits and k-fold cross validation do not work with time series data as they overlook the temporal elements of the issue. The work which this project is based on may also have a similar flaw in how it evaluates the time series. By not respecting the temporal sequence of examples when evaluating models, it allows them to use "future" information to make predictions.
  
  
## Sliding Window Cross Validation
It is a method of model evaluation that is useful when dealing with datasets that have a temporal component. The dataset is split into multiple windows, with each window being used for training and testing a model. The model is then evaluated on the test window, and the results are used to select the best model. The process is then repeated until all windows have been used. This method allows models to be evaluated on data from different points in time, providing a more accurate assessment of model performance. 
  
`<br>
![walking forwad](https://github.com/SimoneParvizi/ML-EEG-eyes-model/assets/75120707/50b279f8-a0e2-4f75-83cf-d769a8b77f5c)
  
  
## Results

The results of the model can be seen in the following table: 

| Metric | Result | 
| --- | --- | 
| F1 Score | 0.9057 | 

The results show that the model is able to predict eye states with an f1 score of 0.9055, which is greater than the desired 90% accuracy.
  
  
## Conclusion

This project presents a Machine Learning model that, given an EEG signal, can predict eye states (i.e. eyes opened or closed) in a time window from 0.1 to 1 second with an f1 score of 0.9055. The model uses a sliding window cross-validation method which is necessary for the model to effectively classify the eye states in the given time window. The findings in this study will contribute to an ideal solution to the problem of EEG based eye state classification.

# For more detailed information download the Project.pdf file
