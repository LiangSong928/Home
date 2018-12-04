
# coding: utf-8

# In[ ]:


import pandas as pd
import Numpy as py
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
train = pd.read_csv('C:/Users/peter/Desktop/Home Default/application_train.csv')

by_contract = train.groupby('NAME_CONTRACT_TYPE')
by_contract_ = test.groupby('NAME_CONTRACT_TYPE')

print(df.train['AMT_INCOME_TOTAL'].count())
print(df.train['AMT_INCOME_TOTAL'].Mean())
print(df.train['AMT_INCOME_TOTAL'].Median())
print(df.train['AMT_INCOME_TOTAL'].Std())

print(df.test['AMT_INCOME_TOTAL'].count())
print(df.test['AMT_INCOME_TOTAL'].Mean())
print(df.test['AMT_INCOME_TOTAL'].Median())
print(df.test['AMT_INCOME_TOTAL'].Std())

Income = df.train['NAME_CONTRACT_TYPE'].sort()
Income_ = df.test['NAME_CONTRACT_TYPE'].sort()

Income1 = df.train['AMT_INCOME_TOTAL'].sort()
Income_1 = df.test['AMT_INCOME_TOTAL'].sort()

print(Income.loc['NAME_CONTRACT_TYPE'])
print(Income_ .loc['NAME_CONTRACT_TYPE'])


#Train/Test Split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = application_train['AMT_INCOME_TOTAL'].data
y = application_train['AMT_CREDIT'].target

#Correlation computation
from scipy.stats.stats import pearsonr
pearsonr(x, y)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

print(df.train.describe())
print(df.test.describe())

#Create Dummy Variable
df_car = pd.get_dummies(df.train['FLAG_OWN_CAR'])


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer 
from sklearn.preprocessing import scale 

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)

X = imp.transform(X) 

# Scaling in a pipeline
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]

pipeline = Pipeline(steps) 
knn_scaled = pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# CossValidation:
# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels,alpha=0.5)

#Dimension Reduction

import array 
arr= array.array(['samples',application_train.loc['AMT_INCOME_TOTAL','AMT_CREDIT']])

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)


# Import scale
from sklearn.preprocessing import scale
# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X))) 
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))









