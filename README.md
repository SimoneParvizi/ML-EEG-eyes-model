
# EEG-Based Eye State Classification

![alt text](image.jpg)

In this project, a Machine Learning (ML) model is presented that can predict eye states (i.e. eyes opened or closed) in a time window from 0.1 to 1 second with over 90% accuracy. The model is trained on an EEG signal.

The model was tested on a dataset of EEG signals and achieved an f1 score ranging from 0.9823 to 0.8208, and an average f1 score of 0.9055. The few predictions where the metric is less than 0.99, match with the transition from eyes opened to closed or vice-versa. 

The findings in this study will contribute to an ideal solution to the problem of EEG based eye state classification. 

## Installation

To install the necessary libraries for this project, run the following command in the terminal:

```
pip install -r requirements.txt
```

## Usage

To use the model, run the following command in the terminal:

```
python main.py
```

## Results

The results of the model can be seen in the following table: 

| Metric | Result | 
| --- | --- | 
| f1 Score | 0.9055 | 

The results show that the model is able to predict eye states with an f1 score of 0.9055, which is greater than the desired 90% accuracy.

## Conclusion

This project presents a Machine Learning model that, given an EEG signal, can predict eye states (i.e. eyes opened or closed) in a time window from 0.1 to 1 second with an f1 score of 0.9055. The findings in this study will contribute to an ideal solution to the problem of EEG based eye state classification.
