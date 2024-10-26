# Завантаження необхідних бібліотек
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Завантаження набору даних
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
         'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
         'hours-per-week', 'native-country', 'income']
dataset = pd.read_csv(url, names=names, na_values="?", skipinitialspace=True)

# Видаляємо пропущені значення
dataset.dropna(inplace=True)

# Перетворюємо категоріальні змінні у числові
dataset = pd.get_dummies(dataset, drop_first=True)

# Розділяємо дані на вхідні (X) та вихідні (Y) змінні
X = dataset.drop('income_>50K', axis=1)
Y = dataset['income_>50K']

# Розділення на тренувальні та тестові вибірки
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Завантажуємо алгоритми моделі
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))

# Оцінюємо кожну модель за допомогою крос-валідації
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})')

# Оцінка якості найкращої моделі на тестовій вибірці
best_model = LinearDiscriminantAnalysis() # Приклад, замініть на модель з найкращим результатом
best_model.fit(X_train, Y_train)
predictions = best_model.predict(X_test)

print(f'Accuracy: {accuracy_score(Y_test, predictions):.4f}')
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
