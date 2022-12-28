import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

#read the data set
df=pd.read_excel("PROJECT\data_given.xlsx")
df['RestaurantID'] = df['RestaurantID'].apply(lambda x: float(x.split()[0].replace('R', '')))
print(len(df))

#Replace zeroes
zero_not_accepted = ['M1DeliveryCost', 'M2ExpectedDeliveryTime', 'M3MinChargeForOrdering','M3LogMinChargeForOrdering','M4NumberOfComments ','M4LogNumberOfComments ','M5DeliveryTimeFulfillment','M6DeliveryCostPerKm']
for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)

df=df.dropna()
# evaluate adaboost algorithm for classification
# define dataset
X = df.iloc[:, 3:19]
y = df.iloc[:, 19]
X, y = make_classification(n_samples=9000, n_features=10, random_state=6)
# define the model
model = AdaBoostClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.5f (%.3f)' % (mean(n_scores), std(n_scores)))

