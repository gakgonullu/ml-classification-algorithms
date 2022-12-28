import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap  
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix  
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
import matplotlib.pyplot as plt

#Scikit Learn Imports
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#read the data set
df=pd.read_excel("PROJECT\data_given.xlsx")
df['RestaurantID'] = df['RestaurantID'].apply(lambda x: float(x.split()[0].replace('R', '')))

#Replace zeroes
zero_not_accepted = ['M1DeliveryCost', 'M2ExpectedDeliveryTime', 'M3MinChargeForOrdering','M3LogMinChargeForOrdering','M4NumberOfComments ','M4LogNumberOfComments ','M5DeliveryTimeFulfillment','M6DeliveryCostPerKm']
for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)


df['M6DeliveryCostPerKm'] = df['M1DeliveryCost'] / df['DistanceKm']

df['M5DeliveryTimeFulfillment'] = df['M2ExpectedDeliveryTime'] - df['TimeMinutes']

df['M3LogMinChargeForOrdering'] = np.log10(df['M3MinChargeForOrdering'])

df=df.dropna()

X = df.iloc[:, 3:19]
y = df.iloc[:, 19]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2000, random_state=0)#test-train split

st_x= StandardScaler()  
X_train= st_x.fit_transform(X_train)    
X_test= st_x.transform(X_test)    
model = MLPRegressor(hidden_layer_sizes=(150,100,50), max_iter = 300,activation = 'relu', solver = 'adam')
model.fit(X_train, y_train)
#Predict on the validation data
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print("Neural Network")

