# This project predicts credicard fraud


# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report,confusion_matrix



data_lnk = 'C:/Users/Gabriel/Desktop/Datasets/creditcard.csv'
df = pd.read_csv(data_lnk)

# Inspect Data
print(df)
print(df.shape)
print(df.info())
print(df.describe())


# Standardization
# Amount and Time need to be standardize
scaler = StandardScaler()
df['Scaled_Amount']=scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['Scaled_Time']=scaler.fit_transform(df['Time'].values.reshape(-1,1))
df=df.drop(['Amount', 'Time'], axis=1)

# Check Class Imbalanace
y = df['Class']
y.hist() # The classes are not balanced
plt.show()

df=df.sample(frac=1)
print(df)

fraud= df.loc[df['Class'] == 1]
non_fraud=df.loc[df['Class']==0][:492]
df = pd.concat([fraud, non_fraud])
data = df.sample(frac=1, random_state=42)
print(data.shape)

sns.countplot(data['Class'], palette='spring')
data['Class'].value_counts()
plt.show()

plt.figure(figsize=(20,14))
sns.heatmap(data.corr(), annot=True, cmap='viridis')
plt.show()

#Train Test Split
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Train the Bagging Classifier

clf = BaggingClassifier(n_estimators=10, random_state=0)
clf.fit(X_train, y_train)

# Predictions and Evaluations
predictions= clf.predict(X_test)
print(confusion_matrix(y_test, predictions))
print('\n')
labels=[0,1]
cmx=confusion_matrix(y_test, predictions, labels)
fig = plt.figure()
ax = fig.add_subplot()
cax = ax.matshow(cmx)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print('\n')
print(classification_report(y_test, predictions))
