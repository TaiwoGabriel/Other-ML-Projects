This project predicts South Africa's COVID-19 cases

import warnings
warnings.filterwarnings('ignore')

# Import Libraries
from numpy import mean,std
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# Importing or Loading dataset
data = "C:/Users/Gabriel/Desktop/sa_data.csv"
df = pd.read_csv(data, delimiter=',')
# Inspect data
print(df.head(10).to_string())
print('Original Data Shape----------:',df.shape, "\n")
# The following features are dropped because they contain no feature values for all records
df = df.drop(["iso_code","continent", "location", "date", "icu_patients", "icu_patients_per_million",
              "hosp_patients", "hosp_patients_per_million", "weekly_icu_admissions", "weekly_icu_admissions_per_million",
              "weekly_hosp_admissions", "weekly_hosp_admissions_per_million", "tests_units",
              "total_boosters","total_boosters_per_hundred", "population", "population_density", "median_age",
              "aged_65_older", "aged_70_older", "gdp_per_capita", "extreme_poverty", "cardiovasc_death_rate",
              "diabetes_prevalence", "female_smokers", "male_smokers", "handwashing_facilities",
              "hospital_beds_per_thousand", "life_expectancy", "human_development_index"], axis=1)

##Check data shape
print('New Data Shape----------:',df.shape)
# DATA PREPARATION--------------------------------------------------------
# Check Data Types
print('Check the Data Types----------')
print(df.info())
print('\n')

# Check Missing Values: Delete columns or rows having missing values more than 30% or to imput values if less--------
miss = df.isnull().sum()
print('Missing values in each feature \n:-------------------------------')
miss2 = df.isnull().sum().sum()
print(miss) # The sonar dataset has no missing values in the features
print("The total number of missing values in the dataset is", miss2)
print("\n")

# Through observation, a number of features has more than 30% missing values in the dataset. Hence, the should be deleted.

# Deleting more features that have more than 30% missing values in them.

df2 = df.drop(["total_vaccinations","people_vaccinated", "people_fully_vaccinated", "new_vaccinations", "new_vaccinations_smoothed",
              "total_vaccinations_per_hundred", "people_vaccinated_per_hundred", "people_fully_vaccinated_per_hundred",
              "new_vaccinations_smoothed_per_million", "new_people_vaccinated_smoothed", "new_people_vaccinated_smoothed_per_hundred",
              "excess_mortality_cumulative_absolute","excess_mortality_cumulative", "excess_mortality",
              "excess_mortality_cumulative_per_million"], axis=1)

print(df2.info())
print('Another new data shape----------:',df2.shape, "\n")

miss_val = df2.isnull().sum()
print('New Missing values in each feature \n:-------------------------------')
miss_val2 = df2.isnull().sum().sum()
print(miss_val) # The sonar dataset has no missing values in the features
print("The new total number of missing values in the dataset is", miss_val2)
print("\n")

# Now we have 22 features that will be used for modelling. The next step is to perform missing value imputation using
# the mean of each features. Mean is used becuase the feature values are numeric as shown by the info()

# Perform missing value imputation using Mean.
df2.fillna(df.mean(),inplace=True)
miss_val3 = df2.isnull().sum()
miss_val4 = df2.isnull().sum().sum()
print(miss_val3) # The sonar dataset has no missing values in the features
print("The new total number of missing values in the dataset is", miss_val4)
print("\n")

# Statistical summary of full dataset shows that there are outliers in almost all features. This is expected because of
# how the feature values were recorded during data collection.
stat = df2.describe()
print(stat.to_string())
print("\n")

# Obtaining the class label: We have to compute the mean value of the positive rate column. Then, if the value of a record in
# the positive_rate column is less than the mean, we replace such value as 0 to denote negative class. On the other hand, if the
# value of a record in the positive_rate column is equal to or greater than the mean, we replace such value as 0 to denote negative class
# Note: It is expected the class computation from the positive rate column might lead to class imbalance in the dataset.
# Therefore, we use SMOTE technique to balance the class, before normalization and splitting the dataset into training and
# test sets respectively.

# statistical summary of positive_rate to obtain the class labels
stat_for_class = df2.positive_rate.describe()
print("Statistical summary for positive_rate to obtain the class labels")
print(stat_for_class)
class_mean = stat_for_class['mean'].round(5)
print("Mean of the positive_rate_column:",class_mean)
print("\n")

#Logic to create the class label column
conditions = [
    (df2['positive_rate'] < class_mean),
    (df2['positive_rate'] >= class_mean)]

# create a list of the values we want to assign for each condition
values = [0, 1]

# create a new column and use np.select to assign values to it using our lists as arguments
df2['class'] = np.select(conditions, values)
# display updated DataFrame
print(df2.head(10).to_string())
print(df2.shape)
print("\n")

# Check the number of class labels in the data
df_class_labels = df2['class'].unique()
print("Class Labels:", df_class_labels)
# NOTE: O represent negative cases of COVID-19, while 1 represent positive cases of COVID-19 disease.

# Class label distribution among samples
class_dist = df2['class'].value_counts()
print("Class distribution:",class_dist)
print("\n")
print("\n")
# The class distribution shows that label 1 (positive is the minority class)

# Feature correlation
df_corr = df2.corr()
print('Feature Correlation Table')
print(df_corr.to_string())
print("\n")
print("Correlation value of each feature to the class label")
df_corr2 = abs(df2.corr())['class'].sort_values(ascending=False)
print(df_corr2)

# From the feature correlation, we can see that the total_deaths, total_deaths_per_million, and reproduction_rate produced
# low correlation to the class. So, they should be deleted.

df3 = df2.drop(["total_deaths", "total_deaths_per_million", "reproduction_rate"], axis=1)
print(df3.info())
print("\n")

# Separate feature vectors from target labels
X = df3.drop(['class'],axis=1)
y = df3['class'].copy()

# Check Class Distribution for Imbalance: random undersampling, SMOTE or ensemble methods (Bagging, Boosting)
# Bagging and SMOTE used for data resampling and to handle the class imbalance problem
# Visualize classes
y.hist()
plt.title('Imbalanced Class Distribution')
plt.show()

# Convert the Dataframe to Numpy Arrays
X = X.values
y = y.values


# Select Relevant features by evaluating feature importance (Dimensionality Reduction)---------------------------
# All features are relevant and used


# DATA PREPARATION ENDS HERE---------------------------------------------------------------------

# Split the dataset into the Training set and Test set-------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Balancing the classes in the training dataset
smt = SMOTE(random_state=42)
X_train, y_train = smt.fit_resample(X_train, y_train)
# Check the shape of the balanced feature vectors
print('Data Shape after balancing')
print('New Feature vector:',X_train.shape)
print('New Class Shape:',y_train.shape)


# Check Class Distribution
print('Class Distribution in training data')
print('Rock:',sum(y_train==0))
print('Mine:',sum(y_train==1))


# Visualize balanced classes
plt.hist(y_train)
plt.title('Balanced Class Distribution ')
plt.show()


# Performing feature normalization or standardization-----------------------------------
# The range of values for the attributes are almost of the same range. Standardization will improve the data
# variability and mean.
scale = StandardScaler() # The standardscaler normalizes the data to a mean of 0 and a standard deviation of 1
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


#View Data after scaling
X_train = pd.DataFrame(X_train)
#print('Scaled Training data')
#print(X_train.head())

X_test = pd.DataFrame(X_test)
#print('Scaled Test data')
#print(X_test.head())


print('\n')
# MODEL DEVELOPMENT BEGINS
print('# MODEL DEVELOPMENT BEGINS')
# Cross validation of 10 folds and 5 runs
cv_method = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)

# Hyperparameter Optimization using RandomSearch and CrossValidation to get the best model hyperparamters

# kNN Classifier
nearest_neighbour = KNeighborsClassifier()
# Create a dictionary of KNN parameters
# K values between 1 and 15 are used to avoid ties and p (representing the distance metric) values of 1 (Manhattan), 2 (Euclidean), and 5 (Minkowski)
param_kNN = {'n_neighbors': [1,3,5,7,9,11,13,15],'p':[1,2,5]} # Distance Metric: Manhattan (p=1), Euclidean (p=2) or
# Minkowski (any p larger than 2). Technically p=1 and p=2 are also Minkowski distances.
# Define the kNN model using RandomSearch and optimize accuracy
kNN_grid = RandomizedSearchCV(nearest_neighbour,param_kNN,scoring='accuracy',cv=cv_method)
kNN_grid.fit(X_train,y_train)
# Print the best parameter values for KNN
print('kNN Best Parameter values =',kNN_grid.best_params_)
kNN = kNN_grid.best_estimator_

# Decision Tree Classifier
Decision_Tree = DecisionTreeClassifier()
# Create a dictionary of DT hyperparameters
params_DT = {'criterion':['gini','entropy'],
             'max_depth':np.arange(1,20),
             'splitter':['best','random']}

# Using Random Search to explore the best parameter for the a decision tree model
DT_Grid = RandomizedSearchCV(Decision_Tree,params_DT,scoring='accuracy',cv=cv_method)
# Fitting the parameterized model
DT_Grid.fit(X_train,y_train)
# Print the best parameter values
print('DT Best Parameter Values:', DT_Grid.best_params_)
DT = DT_Grid.best_estimator_

# Support Vector Machines
SVM_clasf = SVC(probability=True)
# Create a dictionary of SVM hyperparameters
# Parameter space for rbf kernels
params_SVC = {'kernel':['rbf'],'C':np.linspace(0.1,1.0),
              'gamma':['scale','auto']}

# Using Random Search to explore the best parameter for the a SVM model
SVC_Grid = RandomizedSearchCV(SVM_clasf,params_SVC,scoring='accuracy',cv=cv_method)
# Fitting the parameterized model
SVC_Grid.fit(X_train,y_train)
# Print the best parameter values
print('SVC Best Parameter Values:', SVC_Grid.best_params_)
SVC = SVC_Grid.best_estimator_

# Neural Network
mlp = MLPClassifier()
parameter_MLP = {
    'hidden_layer_sizes': [(25,25,25),(50,50,50), (100, 100)],
    'activation': ['relu','tanh'],
    'solver': ['adam'],'max_iter':[500,1000],
    'learning_rate': ['adaptive'],
    'learning_rate_init':[0.0001, 0.001,0.01,0.1,0.5]}

mlp_Grid = RandomizedSearchCV(mlp, parameter_MLP, scoring='accuracy',cv=cv_method)
mlp_Grid.fit(X_train, y_train) # X is train samples and y is the corresponding labels

# Check best hyperparameter and estimator
print('ANN Best parameter values:', mlp_Grid.best_params_)
MLP = mlp_Grid.best_estimator_
print('\n')


# Developing homogeneous ensembles for each classifier
kNN_ensemble = BaggingClassifier(base_estimator=kNN, n_estimators=10)
DT_ensemble = BaggingClassifier(base_estimator=DT, n_estimators=10)
SVM_ensemble = BaggingClassifier(base_estimator=SVC, n_estimators=10)
MLP_ensemble = BaggingClassifier(base_estimator=MLP, n_estimators=10)


# get a list of models to evaluate
def get_models():
    models = dict()
    models['kNN'] = kNN
    models['DT'] = DT
    models['SVM'] = SVC
    models['NN'] = MLP
    models['kNN_ENS'] = kNN_ensemble
    models['DT_ENS'] = DT_ensemble
    models['SVM_ENS'] = SVM_ensemble
    models['NN_ENS'] = MLP_ensemble
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv_method, n_jobs=-1)
    return scores

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
print('Mean Accuracy and Std Dev of each Ensemble on test set:----------------------------------')
for name, model in models.items():
    scores = evaluate_model(model, X_test, y_test)
    results.append(scores)
    names.append(name)
    print('>%s %.3f' % (name, mean(scores)),u"\u00B1", '%.3f' % std(scores))

# plot model performance for comparison
plt.figure()
plt.boxplot(results, labels=names, showfliers=False, showmeans=True)
plt.title('Performance of Ensembles')
plt.xlabel("Ensembles")
plt.ylabel("Accuracy of Ensembles")
plt.show()
#plt.savefig("ensemble_comparison")
print('\n')

print('Mean Accuracy and Std Dev of each Ensemble on train set:-----------------------------')
for name, model in models.items():
    # evaluate the model
    scores = evaluate_model(model, X_train, y_train)
    # store the results
    results.append(scores)
    names.append(name)
    # summarize the performance along the way
    print('>%s %.3f' % (name, mean(scores)), u"\u00B1", '%.3f' % std(scores))


print('\n')

# Train and evaluate each Ensemble
for name,model in models.items():
    # fit the model
    model.fit(X_train,y_train)
    # then predict on the test set
    y_pred= model.predict(X_test)
    # Evaluate the models
    print('Performance Results of', name, ':----------------------------------------------------------')
    clf_report= classification_report(y_test,y_pred)
    # Confusion Matrix: Showing the correctness and misclassifications made my the models
    conf = confusion_matrix(y_test, y_pred)
    print('Classification Report for', name,':')
    print(clf_report)
    print()
    print('Confusion Matrix for',name, ':')
    print(conf)
    print('\n')
