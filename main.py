# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=Warning)

# Section : Data Preprocessing

# i) Importing the dataset
dataset = pd.read_csv('week13_neural_networks/ANN/dataset/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


# ii) Encoding categorical data

# - Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# - One Hot Encoding the "Geography" column
# and the column transfer to read the data more conveniently

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# iii) Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# iv) Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Section 2: Build the ANN

# i) Initializing the ANN
ann = tf.keras.models.Sequential()

# ii) Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# iii) Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# iv) Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Section 3: Training the ANN

# i) Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ii) Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Section 4 - Making the predictions and evaluating the model

# i) Prediction of a single observation

"""
single prediction for the customer with following information: 
Geography: France
CreditScore: 600
Gender: Male
Age: 40
Tenure: 3
Balance: $60000
NumOfProducts: 2
HasCrCard: 1
IsActiveMember: 1
EstimatedSalary: $50000
"""

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

"""
Gives 'False'
So the customer possibly will not CHURN
"""

# ii) Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# iii) Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)

"""
[[1513   82]
[ 196  209]]
"""

# accuracy, precision, recall and f1 scores:
print(classification_report(y_test, y_pred))

"""
              precision    recall  f1-score   support
           0       0.89      0.95      0.92      1595
           1       0.72      0.52      0.60       405
    accuracy                           0.86      2000
   macro avg       0.80      0.73      0.76      2000
weighted avg       0.85      0.86      0.85      2000
"""