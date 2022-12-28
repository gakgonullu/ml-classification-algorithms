import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from sklearn.linear_model import LinearRegression

#read the data set
df=pd.read_excel("PROJECT\data_given.xlsx")

#Replace zeroes
zero_not_accepted = ['M1DeliveryCost', 'M2ExpectedDeliveryTime', 'M3MinChargeForOrdering','M3LogMinChargeForOrdering','M4NumberOfComments ','M4LogNumberOfComments ','M5DeliveryTimeFulfillment','M6DeliveryCostPerKm']
for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)

df=df.dropna()
#Feature selection algortihm (M5Class attribute based)
X = df.iloc[:, 3:17]
y = df.iloc[:, 17]

#KNN applied
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2501, random_state=10)#test-train split
knn = KNeighborsClassifier(n_neighbors=29)
knn.fit(X_train, y_train)
KNeighborsClassifier(n_neighbors=29)

pred= knn.predict(X_test)
print("Test set predictions:\n", pred)
print("Test set score: {:.6f}" .format(np.mean(pred==y_test)))

plt.subplots(figsize=(12,5))
gender_correlation=df.corr()
sns.heatmap(gender_correlation,annot=True,cmap='RdPu')
plt.title('Correlation between the variables')
plt.xticks(rotation=45)
