import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the data
transactions = pd.read_csv('transactions.csv')
print(transactions.head())
print(transactions.info())
print(transactions['amount'].describe())
# This will only filter rows where 'type' equals a specific value, not create a new column
transactions['isPayment'] = transactions['type'].apply(lambda x: 1 if x in ['PAYMENT', 'DEBIT'] else 0)
print(transactions['isPayment'])

transactions['isMovement'] = transactions['type'].apply(lambda x: 1 if x in ['CASH_OUT', 'TRANSFER'] else 0)
print(transactions['isMovement'])
# How many fraudulent transactions?
transactions['accountDiff'] = abs(transactions['oldbalanceOrg'] - transactions['oldbalanceDest'])


# Summary statistics on amount column
features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]
label = transactions['isFraud']

# Create isPayment field
X_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=42)

# Create isMovement field
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(x_test)
# Create accountDiff field
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print(model.coef_)

transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])
transaction4 = np.array([756756.31, 1.0, 0.0, 510025.5])

sample_transactions = np.array([transaction1, transaction2, transaction3, transaction4])

# Combine new transactions into a single array
sample_transactions = scaler.transform(sample_transactions)

sample_transactions = model.predict(sample_transactions)

training_score = model.score(X_train_scaled, y_train)
test_score = model.score(x_test_scaled, y_test)

print(f"Training Score: {training_score}")
print(f"Test Score: {test_score}")