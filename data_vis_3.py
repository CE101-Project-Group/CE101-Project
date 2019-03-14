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

# histogram and normal probability plot of SalePrice
sns.distplot(train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# applying log transformation
train['SalePrice'] = np.log(train['SalePrice'])

# transformed histogram and normal probability plot
sns.distplot(train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# histogram and normal probability plot
sns.distplot(train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)
plt.show()


