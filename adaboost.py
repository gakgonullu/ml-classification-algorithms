import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# test classification dataset
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
df=pd.read_excel("PROJECT\data_given22.xlsx")
df['RestaurantID'] = df['RestaurantID'].apply(lambda x: float(x.split()[0].replace('R', '')))
print(len(df))

#Replace zeroes
zero_not_accepted = ['M1DeliveryCost', 'M2ExpectedDeliveryTime', 'M3MinChargeForOrdering','M3LogMinChargeForOrdering','M4NumberOfComments ','M4LogNumberOfComments ','M5DeliveryTimeFulfillment','M6DeliveryCostPerKm']
for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)

df['M6DeliveryCostPerKm'] = df['M1DeliveryCost'] / df['DistanceKm']
print('Updated DataFrame:')
print(df)

df['M5DeliveryTimeFulfillment'] = df['M2ExpectedDeliveryTime'] - df['TimeMinutes']
print('Updated DataFrame:')
print(df)

df['M3LogMinChargeForOrdering'] = np.log10(df['M3MinChargeForOrdering'])
print('Updated DataFrame:')
print(df)

df=df.dropna()


# define dataset for M5Class
X = df.iloc[:, 3:17]
y = df.iloc[:, 17]

X, y = make_classification(n_samples=9000, n_features=10, random_state=6)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lsvc = LinearSVC(C=0.30, penalty="l2", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape
print(X_new)
X_new.shape
# define the model
model = AdaBoostClassifier()
# evaluate the model
model=model.fit(X_train, y_train)
y_pred = model.predict(X_test)
import sklearn.metrics as metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
accuracy = float(cm.diagonal().sum())/len(y_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("AdaBoost")
print(accuracy)  



