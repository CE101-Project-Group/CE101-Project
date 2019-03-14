# importing the necessary modules to visualise
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from scipy.stats import norm
from scipy import stats
warnings.filterwarnings("ignore")
plt.show()

"""
this will load in the files to train the program, to test the program
and to have a look at some basic mathematical information about the SalePrice
of housing
"""
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train['SalePrice'].describe()
print(train.head(20)) # look at the first 20 entries in the train dataset
print(test.head(20)) # look at the first 20 entries in the test dataset

# drop the id column from the data as they do not have any correlation with the data whatsoever
trainID = train["Id"]
testID = test["Id"]
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# data transformation
train['GrLivArea'] = np.log(train['GrLivArea'])

# transformed histogram and normal probability plot
sns.distplot(train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)

# histogram and normal probability plot
sns.distplot(train['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'], plot=plt)

#transform data
train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

#histogram and normal probability plot
sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
