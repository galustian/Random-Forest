import numpy as np
import pandas as pd
from decisiontree import gini_index, get_best_split
from datetime import datetime

# test gini_index
print(gini_index(np.array([[1.5, 2.1, 3.8, 2.0], [2.0, 4.0, 4.0, 2.0], [3.0, 6.0, 4.0, 3.0]], dtype=np.float64), np.array([[10.0, 12.0, 14.0, 4.0], [12.0, 13.0, 9.0, 4.0], [18.0, 34.0, 12.0, 4.0]], dtype=np.float64)))

# test get_best_split
df = pd.read_csv('iris-test.csv', header=None)
df = df.sample(frac=1, axis=0)
df.iloc[:, -1] = df.iloc[:, -1].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
df_matrix = df.as_matrix()
split_data = get_best_split(df_matrix)
'''print(datetime.now())
for i in range(256):
    split_data = get_best_split(df_matrix)
print(datetime.now())'''

# test ...