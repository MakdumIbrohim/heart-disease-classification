import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

data = pd.read_csv('diabetes.csv')

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols_with_zero:
    data[col] = data[col].replace(0, np.nan)
    data[col].fillna(data[col].median(), inplace=True)

X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

akurasi = accuracy_score(y_test, y_pred)
matriks_konfusi = confusion_matrix(y_test, y_pred)
laporan_klasifikasi = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Akurasi Model :", akurasi)
print("Nilai ROC-AUC :", roc_auc)
print("\nLaporan Klasifikasi:\n", laporan_klasifikasi)

plt.figure(figsize=(6, 5))
sns.heatmap(
    matriks_konfusi,
    annot=True,
    fmt='d',
    cmap='viridis',
    xticklabels=['Normal', 'Diabetes'],
    yticklabels=['Normal', 'Diabetes']
)
plt.title('Matriks Konfusi - Decision Tree')
plt.xlabel('Hasil Prediksi')
plt.ylabel('Data Aktual')
plt.tight_layout()
plt.show()

importance_df = pd.DataFrame({
    'Fitur': X.columns,
    'Tingkat Kepentingan': clf.feature_importances_
}).sort_values(by='Tingkat Kepentingan', ascending=False)

print("\nTingkat Kepentingan Fitur:")
print(importance_df)

plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=['Normal', 'Diabetes'],
    filled=True,
    rounded=True
)
plt.title("Visualisasi Pohon Keputusan")
plt.show()
