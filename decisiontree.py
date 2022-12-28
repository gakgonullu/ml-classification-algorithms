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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix  
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#read the data set
df=pd.read_excel("PROJECT\data_given22.xlsx")
df['RestaurantID'] = df['RestaurantID'].apply(lambda x: float(x.split()[0].replace('R', '')))

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
print(len(df))

print(df)
# split dataset for M5 Class
X = df.iloc[:, 13:17]
y = df.iloc[:, 17]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)#test-train split

st_x= StandardScaler()  
X_train= st_x.fit_transform(X_train)    
X_test= st_x.transform(X_test)    


classifier= DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42)  
classifier.fit(X_train, y_train)  
y_pred= classifier.predict(X_test)  


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
accuracy = float(cm.diagonal().sum())/len(y_test)


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Decision Tree")
print(accuracy)













