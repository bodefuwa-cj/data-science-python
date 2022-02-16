"""
Buy a Vineyard Assignment (Part 1)
----------------------------------
Introduction
------------
As Mr Mahi’s wealth grows he has taken interest in the more finer things in life. Wine is one of them.
He enjoys drinking wine so much that he has decided to buy a vineyard. He currently is looking at
different vineyards that are for sale and will decide to purchase one.
This assignment will have 2 parts. One is due this week and one is due next week.
To decide which vineyard to purchase we need to know the price, size and quality of the wine that it
produces. The size and price are easy to figure out. Quality is another story. Figuring out the quality of
wine is not easy.
Here is what we are going to do. We have a wine data set. In the data set we know 11 attributes about
almost 5,000 wines. These are easy to measure, like pH level. There is a simple test for that. The people
that made the data set hired experience wine judges to judge each wine. Their quality score (from 0 to
10) is in column 12.
Fixed acidity
Volatile acidity
Citric acid
Residual sugar
Chlorides
Free sulfur dioxide
Total sulfur dioxide
Density
pH
Sulphates
Alcohol
Quality – this is the value that we are trying to predict.
Before we try to create a wine classification solution, we see a problem with the data. The values are very
different ranges and scales. For example Chlorides is normally less than 0.05, while Total sulfur dioxide
is normally over 100.
You and Mai decided that you need some pre-processing of the data. For each column you would like the
minimum value to scale to 0 and the maximum value scale to 1. For example, let’s say that the values in
a column of data are [0, 2, 3, 4, 5, 10]. 0 is the smallest, that stays 0. 10 is the largest, that becomes 1.
Your scaled data would be [0, .2, .3, .4, .5, 1.0].
Instructions
------------
Create a Python program to load the wine quality data file. Scale all of the columns to be scaled values
between 0 and 1. Then print a few lines of the original data, followed by the scaled data. You must use
Scikit Learn; you cannot just hand write a bunch of Python code to do it.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
# Load data
winedata = np.loadtxt('winedata.csv', delimiter=',', skiprows=1)
DomainVal = winedata[:,0:12]
RangeVal = winedata[:,11:12]
x_train, x_test, y_train, y_test = train_test_split(DomainVal, RangeVal, test_size= 0.25, random_state=42)

# Scale data
objectss= StandardScaler()
objectmm = MinMaxScaler()
X_train_ssscaled = objectss.fit_transform(x_train)
objectmm.fit(x_train)
X_train_mmscaled = objectmm.transform(x_train)
X_test_ssscaled = objectss.fit_transform(x_test)
objectmm.fit(x_test)
X_test_mmscaled = objectmm.transform(x_test)

# Print
print(x_test[0:4,0:12])
print(X_test_ssscaled[0:4,0:12]) 
print(X_test_mmscaled[0:4,0:12])

