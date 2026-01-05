import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('heart.csv')
data.head()

x = data.drop(['target'], axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'Akurasi: {accuracy_score(y_test, y_pred)}\n')
print(f'Laporan Klasifikasi:\n{classification_report(y_test, y_pred)}\n')
print(f'Matriks Kebingungan:\n{confusion_matrix(y_test, y_pred)}')