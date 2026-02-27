# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: PARAVEZHAA M
RegisterNumber: 212225220070

import pandas as pd
data = pd.read_csv("Employee.csv")
print("data.head():")
print(data.head())

print("data.info():")
print(data.info())

print("isnull() and sum():")
print(data.isnull().sum())

print("data value counts():")
print(data["left"].value_counts())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

print("Encoded Salary Head:")
print(data.head())

x = data[["satisfaction_level",
          "last_evaluation",
          "number_project",
          "average_montly_hours",
          "time_spend_company",
          "Work_accident",
          "promotion_last_5years",
          "salary"]]

print("x.head():")
print(x.head())

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy value:")
print(accuracy)
print("Data Prediction:")
print(dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]]))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=x.columns, class_names=["Stayed", "Left"], filled=True)
plt.show()
 
*/
```

## Output:
<img width="1491" height="560" alt="Screenshot 2026-02-27 140726" src="https://github.com/user-attachments/assets/c56e311f-00ae-48b9-8466-a26ddf8ec556" />
<img width="1560" height="702" alt="Screenshot 2026-02-27 140915" src="https://github.com/user-attachments/assets/325e2139-3540-478d-8c2e-df6c9cffde5e" />
<img width="1553" height="707" alt="Screenshot 2026-02-27 140954" src="https://github.com/user-attachments/assets/fb37a82c-bc61-4246-908f-97c21580de25" />
<img width="1553" height="707" alt="Screenshot 2026-02-27 140954" src="https://github.com/user-attachments/assets/8c5c6c39-f40f-456c-9555-d8f1c01225d4" />
<img width="1420" height="760" alt="Screenshot 2026-02-27 141102" src="https://github.com/user-attachments/assets/6875b5cd-59fa-42d8-8853-ca2305e0f611" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
