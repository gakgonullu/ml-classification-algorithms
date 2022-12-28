import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
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
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif

#read the data set
df=pd.read_excel("PROJECT\data_given.xlsx")
df['RestaurantID'] = df['RestaurantID'].apply(lambda x: float(x.split()[0].replace('R', '')))
print(len(df))


#Replace zeroes (mean without zeroes -> skipna=True)
zero_not_accepted = ['M1DeliveryCost', 'M2ExpectedDeliveryTime', 'M3MinChargeForOrdering','M3LogMinChargeForOrdering', 'M5DeliveryTimeFulfillment', 'TimeMinutes', 'RestaurantLatitude']
for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)

#dropna for meaningfull 0 columns "M4Comment, M5(binary)"
special=['M6DeliveryCostPerKm']
for column in special:
    df[column] = df[column].replace(np.NaN, 0)

#Updating M6, M5, M3
df['M6DeliveryCostPerKm'] = df['M1DeliveryCost'] / df['DistanceKm']
print('Updated DataFrame:')
print(df)

df['M5DeliveryTimeFulfillment'] = df['M2ExpectedDeliveryTime'] - df['TimeMinutes']
print('Updated DataFrame:')
print(df)

df['M3LogMinChargeForOrdering'] = np.log10(df['M3MinChargeForOrdering'])
print('Updated DataFrame:')
print(df)

df.loc[(df['TimeMinutes'] > 45) & (df['RestaurantLatitude'] > 4.69), 'M5Class'] = 0


print(df.iloc[0:10, 16:19]) #controlling the updates between line 41 and line 52
df=df.dropna()

# split dataset (M6Class based)
X = df.iloc[:, 3:18]
y = df.iloc[:, 19]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3000, random_state=42)#test-train split

st_x= StandardScaler()  
X_train= st_x.fit_transform(X_train)    
X_test= st_x.transform(X_test)    

model = GaussianNB()

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("Gaussian score is:",score)

