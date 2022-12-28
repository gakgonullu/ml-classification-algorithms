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
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif
from sklearn.linear_model import LinearRegression

#read the data set
df=pd.read_excel("data_given.xlsx")

#Replace zeroes
zero_not_accepted = ['M1DeliveryCost', 'M2ExpectedDeliveryTime', 'M3MinChargeForOrdering','M3LogMinChargeForOrdering','M5DeliveryTimeFulfillment']
for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)


#dropna for meaningfull 0 columns "M4Comment, M5(binary)"

special=['M6DeliveryCostPerKm']
for column in special:
    df[column] = df[column].replace(np.NaN, 0)

#Updating M6, m4log
df['M6DeliveryCostPerKm'] = df['M1DeliveryCost'] / df['DistanceKm']
print('Updated DataFrame:')
print(df)

df['M5DeliveryTimeFulfillment'] = df['M2ExpectedDeliveryTime'] - df['TimeMinutes']
print('Updated DataFrame:')
print(df)

import math
df['M3LogMinChargeForOrdering'] = np.log10(df['M3MinChargeForOrdering'])
print('Updated DataFrame:')
print(df)


print(df.iloc[0:10, 16:18])

df=df.dropna()

#splitting test and train datasets (M5 based)
X = df.iloc[:, 3:17]
y = df.iloc[:, 17]

print(df.iloc[0:10, 16:18])


#KNN applied
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2000, random_state=10)#test-train split
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train, y_train)
KNeighborsClassifier(n_neighbors=21)

pred= knn.predict(X_test)
print("Test set predictions:\n", pred)
print("Test set score: {:.6f}" .format(np.mean(pred==y_test)))

plt.subplots(figsize=(15,7))
delivery_correlation=df.corr()
sns.heatmap(delivery_correlation,annot=True,cmap='RdPu')
plt.title('Correlation between the variables')
plt.xticks(rotation=45)