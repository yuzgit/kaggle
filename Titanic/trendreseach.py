import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

## 訓練データ、テストデータの定義
train = pd.read_csv('datasets/train.csv')
test  = pd.read_csv('datasets/test.csv')

print(train.columns)