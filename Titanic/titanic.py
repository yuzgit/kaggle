import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

## 訓練データ、テストデータの定義
train = pd.read_csv('datasets/train.csv')
test  = pd.read_csv('datasets/test.csv')

## データ抜けを数える
# print(pd.isnull(train).sum())

## 性別特徴
# sns.barplot(x='Sex', y='Survived', data=train)
# plt.show()

##  Plcass(チケットクラス)特徴
# sns.barplot(x='Pclass', y='Survived', data=train)
# plt.show()

## SibSp(同乗している兄弟・配偶者数)特徴
# sns.barplot(x='SibSp', y='Survived', data=train)
# plt.show()

## Parch(同乗している親、子供の数)特徴
# sns.barplot(x='Parch', y='Survived', data=train)
# plt.show()

## Age特徴(0-100歳とかあまり幅広すぎるとわかりづらいので再設定し可視化)
# Ageは欠損率が高いので欠損部分はfillnaで-0.5で埋める
train['Age'] = train['Age'].fillna(-0.5)
test['Age']  = test['Age'].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Tennager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
test['AgeGroup']  = pd.cut(test['Age'], bins, labels = labels)

# sns.barplot(x='AgeGroup', y='Survived', data=train)
# plt.show()

## Cabin特徴
train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

# #calculate percentages of CabinBool vs. survived
# print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)

# print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)
# #draw a bar plot of CabinBool vs. survival
# sns.barplot(x="CabinBool", y="Survived", data=train)
# plt.show()

## データクリーニング
## Cabin クリーニング
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)

## Ticket クリーニング
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)

## Embarked クリーニング (2件欠損値があるので多数派のSで置き換え)
train = train.fillna({"Embarked": "S"})

## Age クリーニング (177件欠損があるので別カラムを作って元のAgeカラムは削除)
combine = [train, test]

for dataset in combine:
    # '英字.'の部分(Mr. Ms.など)を切り出して'Title'とする
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for dataset in combine:
    # 不要なTitleはRareで置換する
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    # その他表現の散らばりも補正する
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    # 扱いやすいようTitleを数字でマッピングする
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# マッピングされたTitle内の最頻値をmodeで取得 (MrならYoung Adultがもっとも多い)
mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

# Age Groupの値がない場合TitleからAgeGroupの値を導き代入 (TitleがStudentなら2をAgeGroupへ代入)
for x in range(len(train["AgeGroup"])):
    # add
    train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
        
for x in range(len(test["AgeGroup"])):
    # add
    test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]


age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

# AgeGroupも扱いやすいよう数字でマッピング
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

# 不要になったAgeカラムをドロップ
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)

## Name クリーニング
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)

## Sex クリーニング 扱いやすいよう数字でマッピング
sex_mapping = {"male" : 0, "female" : 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

## Embarked クリーニング 扱いやすいよう数字でマッピング
embarked_mapping = {"S" : 1, "C" : 2, "Q" : 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

## Fare クリーニング
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


## モデルの選択
# モデル構築用のテストデータを作る、現在の訓練データの22%をテストデータにする
from sklearn.model_selection import train_test_split

# 原因不明のNaNが出るので一旦AgeGroupは除外
predictors = train.drop(['Survived', 'PassengerId'], axis=1)
print(pd.isnull(predictors).sum())
terget = train['Survived']

x_train, x_val, y_train, y_val = train_test_split(predictors, terget, test_size = 0.22, random_state = 0)

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)

#set ids as PassengerId and predict survival 
ids = test['PassengerId']

predictions = gbk.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)