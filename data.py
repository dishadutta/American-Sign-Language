import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

dir_path = "Dataset"
for dirname, _, filenames in os.walk(dir_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv('Dataset/sign_mnist_train.csv')
test = pd.read_csv('Dataset/sign_mnist_test.csv')

print(train.shape)
print(test.shape)

train.head()

train_set = np.array(train, dtype = 'float32')
test_set = np.array(test, dtype='float32')

np.save('train_set.npy', train)
np.save('test_set.npy', test)
