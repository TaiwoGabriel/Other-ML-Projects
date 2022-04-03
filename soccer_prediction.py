import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import pickle
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.model_selection import train_test_split

# Importing or Loading dataset
data = "C:/Users/Gabriel/Desktop/full_dataset2.csv"
df = pd.read_csv(data, delimiter=',')

def data_transform():
    # feature selection
    new_df = df.drop(df.iloc[:, 0:23], axis=1,
                     inplace=False)  # Drop a number of categorical features after performing feature correlation
    new_df = new_df.drop(['type'], axis=1, inplace=False)  # Drop the type column
    new_df = new_df.dropna(axis=0, inplace=False)  # Drop any row with missing values

    # Obtain target class from the score column
    new_df['outcome'] = new_df['score_fulltime'].apply(lambda i: 1 if i[0] > i[-1] else (2 if i[0] < i[-1] else 0))
    new_df = new_df.drop(['score_fulltime'], axis=1, inplace=False)  # Drop the score_fulltime column

    # Removing % sign
    columns_to_check = new_df.columns
    new_df[columns_to_check] = new_df[columns_to_check].replace({r'\%': ''}, regex=True)

    # Converting the feature values with % to floating values
    new_df.iloc[:, 0:21] = new_df.iloc[:, 0:21].astype('float') / 100.0

    # Remove more rows with missing values
    new_df = new_df.dropna(axis=0, inplace=False)
    return new_df



X = data_transform().drop('outcome',axis=1)
y = data_transform()['outcome'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scale = MinMaxScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


print('Cross Validation')
# Cross validation of 10 folds and 5 runs
cv_method = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)

# Support Vector Machines
SVM_clasf = SVC(probability=True)
# Create a dictionary of SVM hyperparameters
# Parameter space for rbf kernels
params_SVC = {'kernel':['rbf'],'C':np.linspace(0.1,1.0),
              'gamma':['auto']}

# Using Random Search to explore the best parameter for the a SVM model
SVC_Grid = RandomizedSearchCV(SVM_clasf,params_SVC,scoring='accuracy',cv=cv_method)
# Fitting the parameterized model
SVC_Grid.fit(X_train,y_train)
# Print the best parameter values
print('SVC Best Parameter Values:', SVC_Grid.best_params_)
SVC = SVC_Grid.best_estimator_
print('\n')



# GradientClassifier
GBC = GradientBoostingClassifier()
# Create a dictionary of SVM hyperparameters
# Parameter space for rbf kernels
params_GBC = {'n_estimators':np.arange(100,350,50),'learning_rate':np.linspace(0.001,0.1),
              'max_depth': np.arange(2,21)}

# Using Random Search to explore the best parameter for the a SVM model
GBC_Grid = RandomizedSearchCV(GBC,params_GBC,scoring='accuracy',cv=cv_method, n_jobs=-1)
# Fitting the parameterized model
GBC_Grid.fit(X_train,y_train)
# Print the best parameter values
print('GB Best Parameter Values:', GBC_Grid.best_params_)
GB = GBC_Grid.best_estimator_
print('\n')



# XGBClassifier
XGBC = XGBClassifier(n_jobs=-1, booster='gbtree')
# Create a dictionary of SVM hyperparameters
# Parameter space for rbf kernels
params_XGBC = {'n_estimators':np.arange(100,350,50),'eta':[0.0001, 0.001, 0.01, 0.1, 1.0],
              'max_depth': np.arange(2,21)}

# Using Random Search to explore the best parameter for the a SVM model
XGBC_Grid = RandomizedSearchCV(XGBC,params_XGBC,scoring='accuracy',cv=cv_method, n_jobs=-1)
# Fitting the parameterized model
XGBC_Grid.fit(X_train,y_train)
# Print the best parameter values
print('XGB Best Parameter Values:', XGBC_Grid.best_params_)
XGB = XGBC_Grid.best_estimator_
print('\n')



# LGBMClassifier
LGBMC = LGBMClassifier(max_depth=-1, n_jobs=-1)
# Create a dictionary of SVM hyperparameters
# Parameter space for rbf kernels
params_LGBMC = {'n_estimators': np.range(100,350,50),'learning_rate':[0.0001, 0.001, 0.01, 0.1, 1.0],
                'boosting_type':['gbdt', 'rf']}

# Using Random Search to explore the best parameter for the a SVM model
LGBMC_Grid = RandomizedSearchCV(LGBMC,params_LGBMC,scoring='accuracy',cv=cv_method, n_jobs=-1)
# Fitting the parameterized model
LGBMC_Grid.fit(X_train,y_train)
# Print the best parameter values
print('LGBM Best Parameter Values:', LGBMC_Grid.best_params_)
LGBM = LGBMC_Grid.best_estimator_
print('\n')
