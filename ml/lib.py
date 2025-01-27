import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    "Student_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [35, 50, 65, 70, 85, 90, 45, 55, 60, 75],
    "Pass": [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]  # 0 = Fail, 1 = Pass
}
df = pd.DataFrame(data)

X = df[['Marks']]  
y = df['Pass']     

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Predictions for Test Data:\n", pd.DataFrame({'Marks': X_test['Marks'], 'Actual': y_test, 'Predicted': y_pred}))