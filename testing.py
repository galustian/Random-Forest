import numpy as np
import pandas as pd
from randomforest import gini_index, get_best_split, create_forest
from datetime import datetime
from tools import plotting

# test gini_index
print(gini_index(np.array([[1.5, 2.1, 3.8, 2.0], [2.0, 4.0, 4.0, 2.0], [3.0, 6.0, 4.0, 3.0]], dtype=np.float64), np.array([[10.0, 12.0, 14.0, 4.0], [12.0, 13.0, 9.0, 4.0], [18.0, 34.0, 12.0, 4.0]], dtype=np.float64)))

# test get_best_split
df = pd.read_csv('iris-test.csv', header=None)
df = df.sample(frac=1, axis=0)
df.iloc[:, -1] = df.iloc[:, -1].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
split_data = get_best_split(df.as_matrix())
#print(split_data)

print(datetime.now())
for i in range(25):
    split_data = get_best_split(df.as_matrix())
print(datetime.now())

# test create_tree / recurse_tree
#print(create_tree(df.as_matrix(), n_trees=2, max_depth=3, max_x=5))
train_data, test_data = df.iloc[:120, [3, 4, -1]].as_matrix(), df.iloc[120:, [3, 4, -1]].as_matrix()
random_forest = create_forest(train_data, n_trees=450, max_depth=15, max_x=3)
predictions = random_forest.predict(test_data)

correct = 0
for i in range(len(predictions)):
    if predictions[i] == df.iloc[120+i, -1]:
        correct += 1
    
print(predictions)
print("Ratio Correct:", correct / len(predictions))


plotting.plot_decision_region(test_data, test_data[:, -1], model=random_forest)