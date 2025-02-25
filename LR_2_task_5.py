# ===================================================
# Приклад класифікатора Ridge #
# ======================================================================
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split  # Додано імпорт для train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from io import BytesIO  # neded for plot
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# Завантаження набору даних Iris
iris = load_iris()
X, y = iris.data, iris.target

# Розподіл даних на навчальну та тестову вибірки
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)

# Створення класифікатора Ridge
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(Xtrain, ytrain)

# Прогнозування на тестовій вибірці
ypred = clf.predict(Xtest)

# Виведення показників якості
print('Accuracy:', np.round(metrics.accuracy_score(ytest, ypred), 4))
print('Precision:', np.round(metrics.precision_score(ytest, ypred, average='weighted'), 4))
print('Recall:', np.round(metrics.recall_score(ytest, ypred, average='weighted'), 4))
print('F1 Score:', np.round(metrics.f1_score(ytest, ypred, average='weighted'), 4))
print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(ytest, ypred), 4))
print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(ytest, ypred), 4))

# Звіт про класифікацію
print('\t\tClassification Report:\n', metrics.classification_report(ytest, ypred))

# Матриця плутанини
mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.title('Confusion Matrix')
plt.savefig("Confusion.jpg")
plt.show()
