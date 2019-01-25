import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Importing dataset and defining independent variables(x) and dependent variable(y)
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
x[:, 1] = labelencoder_x1.fit_transform(x[:, 1])
labelencoder_x2 = LabelEncoder()
x[:, 2] = labelencoder_x2.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features= [1])
x = onehotencoder.fit_transform(x).toarray()
# to avoid dummy variable trap
x = x[:, 1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# initializing the ANN
classifier = Sequential()

# Adding input layer and the first hidden layer
classifier.add(Dense(input_dim = 11, kernel_initializer='glorot_uniform', activation='relu', units = 6))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer='glorot_uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer='glorot_uniform', activation='sigmoid'))

# compiling the ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(x_train, y_train, batch_size = 100, epochs = 100)

# Making predictions on test set
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing the ANN
from ann_visualizer.visualize import ann_viz
ann_viz(classifier,title = 'Bank churn rate')
