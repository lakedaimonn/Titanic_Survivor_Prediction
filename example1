import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import tensorflow as tf
tf.random.set_seed(777) # tf.set_random_seed doesn't work
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

train_df = pd.read_excel('/home/titanic_wiki.xlsx', sheet_name = 'train')
test_df = pd.read_excel('/home/titanic_wiki.xlsx', sheet_name = 'test')

labels = ['death', 'survival']

# delete unnecessary columns
x_train_df = train_df.drop(['name', 'ticket', 'survival'], axis = 1)
x_test_df = test_df.drop(['name', 'ticket', 'survival'], axis = 1) 
y_train_df = train_df[['survival']] 
y_test_df = test_df[['survival']]

print(x_train_df.head()) 
'''
  pclass     sex  age  sibsp  parch   fare embarked
0       2  Female   17      0      0  12.00        C
1       3  Female   37      0      0   9.59        S
2       3    Male   18      1      1  20.21        S
3       3    Male   30      0      0   7.90        S
4       3    Male   25      0      0   7.65        S
'''

print(x_train_df.columns)
'''
Index(['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'], dtype='object')
'''

transformer = make_column_transformer(
    (OneHotEncoder(), ['pclass', 'sex', 'embarked']),
    remainder = 'passthrough')
transformer.fit(x_train_df)
x_train = transformer.transform(x_train_df)
x_test = transformer.transform(x_test_df)

y_train = y_train_df.values
y_test = y_test_df.values

print(x_train.shape)
print(y_train.shape)
'''
   pclass     sex  age  sibsp  parch   fare embarked
0       2  Female   17      0      0  12.00        C
1       3  Female   37      0      0   9.59        S
2       3    Male   18      1      1  20.21        S
3       3    Male   30      0      0   7.90        S
4       3    Male   25      0      0   7.65        S
Index(['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'], dtype='object')
(730, 12)
(730, 1)
'''

input = Input(shape = (12,))
net = Dense(units = 512, activation = 'relu')(input)
net = Dense(units = 512, activation = 'relu')(net)
net = Dense(units = 1, activation = 'sigmoid')(input)
model = Model(inputs = input, outputs = net)

model.summary()
'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 12)]              0
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 13
=================================================================
Total params: 13
Trainable params: 13
Non-trainable params: 0
_________________________________________________________________
'''

from keras.callbacks import TensorBoard

model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.01), metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test),
callbacks=[TensorBoard(log_dir='./tensorboard/titanic_survival_classification_model')])

X_test = transformer.transform(pd.DataFrame([[2, 'Female', 21, 0, 1, 21.00, 'S']],
columns = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']))
print(X_test)
'''
[[ 0.  1.  0.  1.  0.  0.  0.  1. 21.  0.  1. 21.]]
'''

y_predict = model.predict(x_test)
print(y_predict)
print(y_predict.flatten())
print(y_predict.flatten()[0])
print(1 if y_predict.flatten()[0] > 0.5 else 0)
print(labels[1 if y_predict.flatten()[0] > 0.5 else 0])
