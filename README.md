![churn_magnet](https://github.com/sayinumutsayin/Churn_prediction_with_ANN/assets/146718157/8d46ac46-9d6c-41aa-8bd6-12e60f02a38d)

# Sometimes the customers leave their banks for several types of reasons such as ⚠️poor customer services, ⚠️high fees, ⚠️better offers from competitors etc. 
# Each churn means that the bank will loose money because their customers are the ones who buy products from the bank. 
# To avoid customer churn, banks take some precautions; such as ❇️offering the customers some campaigns, ❇️reducing the annual credit card fees etc. 
# But of course, to be able to do this, they need to *️⃣identify the customers that might leave their bank so that they can offer campaigns to these spesific customers. 
# ➡️ In this project, a churn prediction model -an 🔸Artificial Neural Network🔸 model- is used with the following accuracy, recall, precision and f1 scores;

# 🔴 1: churn
# 🟢 0: stays

<img width="882" alt="precision_recall_accuracy" src="https://github.com/sayinumutsayin/Churn_prediction_with_ANN/assets/146718157/c0e16f8f-5d42-46b8-83a3-7fb4b7b3fad8">


# 🔵 Problem:

# The banks need to know if their customer will possibly leave their bank or no. To be able to do this, they will use the previous data to make the predictions for the future. Afterwards, they will be able to focus on the customers that will possibly be churn.

----------------------------------------------------------------------------------------------------
  
# 🔵 The 14 features of the dataset:

# 🔸 RowNumber: Index starting from 1
# 🔸 CustomerId: Unique id of each customer
# 🔸 Surname: Surname if each customer
# 🔸 CreditScore: Credit score of each customer calculated by the bank
# 🔸 Geography: Countries of the customers (France,Germany, Spain)
# 🔸 Gender: Gender of each customer
# 🔸 Age: Age of each customer
# 🔸 Tenure: The duration passed since the person became a customer (year)
# 🔸 Balance: The balance(money) of the customer in his/her bank account
# 🔸 NumOfProducts: The number of products that the customer have in the bank such as credits, retirement plans etc
# 🔸 HasCrCard: If the customer has a credit card from the bank. (1, 0)
# 🔸 IsActiveMember: Is the customer actively making transactions (1, 0)
# 🔸 EstimatedSalary: Estimated salary of the customer
# 🔸 Exited (dependent variable): If the customer is still in the bank or churned (1 = churned, 0 = still customer)

----------------------------------------------------------------------------------------------------




# 🔵 The project consists of 4 Sections:
  
# Section 1 - Data Preprocessing:

✅ Importing the dataset

✅ Encoding categorical data: LabelEncoder and OneHotEncoder

✅ Splitting the dataset into the Training set and Test set

✅ Feature Scaling: StandardScaler


----------------------------------------------------------------------------------------------------


# Section 2 - Building the ANN:
 
✅ Initializing the ANN

✅ Adding the input layer and the first hidden layer

✅ Adding the second hidden layer

✅ Adding the output layer


----------------------------------------------------------------------------------------------------


# Section 3 - Training the ANN:

✅ Compiling the ANN

✅ Training the ANN on the Training set

  
----------------------------------------------------------------------------------------------------


# Section 4 - Making the predictions and evaluating the model:

✅ Predicting the result of a single observation

✅ Predicting the Test Set results

✅ Making the Confusion Matrix and checking the precision, recall and f1 scores

----------------------------------------------------------------------------------------------------


# 🔵 The following Python libraries were used during the project:

▶ Numpy

▶ Pandas

▶ TensorFlow

▶ Scikit-learn


