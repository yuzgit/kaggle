# 参考URL: https://www.kaggle.com/omarelgabry/a-journey-through-titanic

## imports ##
# pandas
import pandas as pd
from pandas import Series, DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

## データセット読み込み
titanic_df = pd.read_csv('datasets/train.csv')
test_df    = pd.read_csv('datasets/test.csv')

# datainfo
print(titanic_df.describe())
print('----------------------------')
print(test_df.describe())