import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,plot_confusion_matrix


#read the data set
df=pd.read_excel("PROJECT\data_given.xlsx")

#Replace zeroes
zero_not_accepted = ['M1DeliveryCost', 'M2ExpectedDeliveryTime', 'M3MinChargeForOrdering','M3LogMinChargeForOrdering','M4NumberOfComments ','M4LogNumberOfComments ','M5DeliveryTimeFulfillment','M6DeliveryCostPerKm']
for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)

df=df.dropna()

#Feature selection algortihm
X = df.iloc[:, 3:18]
y = df.iloc[:, 18]

#KNN applied
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2501, random_state=10)#test-train split

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svc = SVC(kernel='linear', C=20.0, random_state=1)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

print('Recall: %.3f' % recall_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))