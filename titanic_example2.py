import pandas as pd
import numpy as np

data_train = pd.read_csv('/home/train.csv')
data_test = pd.read_csv('/home/test.csv')
data_check = pd.read_csv('/home/test_result.csv')

# Change column Name
data_train = data_train.rename(columns = {'Pclass' : 'TicketClass'})
data_test = data_test.rename(columns = {'Pclass' : 'TicketClass'})

# Remove unused columns
data_train = data_train.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Age'], axis = 1)
data_test = data_test.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1)

# importing LabelEncoder from Sklearn
from sklearn.preprocessing import LabelEncoder
label_encoder_sex = LabelEncoder()

# use Slicing to select only Sex column from datatframe
# transforming Sex column values using LabelEncoder
data_train.iloc[:,3] = label_encoder_sex.fit_transform(data_train.iloc[:,3])
data_test.iloc[:,2] = label_encoder_sex.fit_transform(data_test.iloc[:,2])

# arranging columns for the Dataframe
data_train = data_train[['PassengerId', 'Sex', 'SibSp', 'Parch', 'TicketClass', 'Survived']]
data_test = data_test[['PassengerId', 'Sex', 'SibSp', 'Parch', 'TicketClass']]
# print(data_train)

# colum 0-4 should be input and 5th column should be output
x_train = data_train.iloc[:, 0:5] # Inputs # 0:5 -> up to but not including 5
y_train = data_train.iloc[:, 5] # output(Survived)

# Use StandardScaler to normalize our data*****
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train) 
X_test = sc.fit_transform(data_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
# The Sequential model is a linear stack of layers.
# The model needs to know what input shape it should expect. For this reason, the first layer in a Sequential model (and only the first, because following layers can do automatic shape inference) needs to receive information about its input shape. There are several possible ways to do this:
# Some 2D layers, such as Dense, support the specification of their input shape via the argument input_dim,
# Some 2D layers, such as Dense, support the specification of their input shape via the argument input_dim,

# Now create object of the Sequential class
classifier = Sequential()

# adding layers to our classifier model, we are building a neural network with 3 layers
# Input layer with 5 input neurons
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 5))
# Hidden layer
classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu'))
# output layer with 1 output neuron which will predic 1 or 0
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# we have added layers to our model, now it's time to compile it

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# now provide training data to our model so that it can Sklearn
# use epoch = 100, so it will not overfit
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# our model is trained. Let it predict on a new situation
# getting predictions of test data
prediction = classifier.predict(X_test).tolist()
# list to series
se = pd.Series(prediction) ### what is series??????
#create new column of predictions in data_check dataframe(.csv file)
data_check['check'] = se
data_check['check'] = data_check['check'].str.get(0) 


# values greater than 0.5 in column 'check' are predicted as 1
# let's write a loop to add another column in which value >=0.5 has 0
series = []
for val in data_check.check:
    if val >= 0.5:
        series.append(1)
    else:
        series.append(0)
# print(data_check)

data_check['final'] = series
# add column named 'final' inside test_result.csv and append either 0 or 1

# let's write another loop to count how many predictions were right
# typical counting loop
match = 0
nomatch = 0
for val in data_check.values:
    if val[1] == val[3]: 
        match = match + 1
    else:
        nomatch = nomatch + 1
print('match:', match, 'nomatc:', nomatch)
