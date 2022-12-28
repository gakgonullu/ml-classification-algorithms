import numpy as np
import pandas as pd

from sklearn.model_selection import LeavePOut,cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 



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

df=df.dropna()

X = df.iloc[:, 3:17]
y = df.iloc[:, 17]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=2000)#test-train split


st_x= StandardScaler()  
X_train= st_x.fit_transform(X_train)    
X_test= st_x.transform(X_test)    

# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 50, max_depth=2, random_state=20)  
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn import metrics  
print()
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
y_pred= clf.predict(X_test)  


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
accuracy = float(cm.diagonal().sum())/len(y_test)


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Random Forest")
print(accuracy)