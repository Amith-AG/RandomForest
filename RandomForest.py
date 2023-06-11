import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the diabetes dataset
diabetes = pd.read_csv('diabetes.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.loc[:, diabetes.columns != 'Outcome'], 
    diabetes['Outcome'], 
    stratify=diabetes['Outcome'], 
    random_state=66
)

# Create an instance of Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)

# Fit the Random Forest model
rf.fit(X_train, y_train)

# Print accuracy on training set
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))

# Print accuracy on test set
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))
