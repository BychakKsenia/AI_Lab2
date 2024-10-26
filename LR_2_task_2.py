import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Завантаження та підготовка даних, як у попередньому завданні
input_file = 'income data.txt'
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoder.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Словник для збереження результатів метрик
results = {}

# Перелік ядер для тестування
kernels = ['poly', 'rbf', 'sigmoid']
for kernel in kernels:
    # Створення та навчання SVM-класифікатора з різними ядрами
    classifier = SVC(kernel=kernel, random_state=0)
    classifier.fit(X_train, y_train)

    # Передбачення для тестових даних
    y_test_pred = classifier.predict(X_test)

    # Обчислення метрик для поточного ядра
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Збереження результатів
    results[kernel] = {
        'Accuracy': round(100 * accuracy, 2),
        'Precision': round(100 * precision, 2),
        'Recall': round(100 * recall, 2),
        'F1 Score': round(100 * f1, 2)
    }

# Виведення результатів для кожного ядра
for kernel, metrics in results.items():
    print(f"Kernel: {kernel}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}%")
    print("-" * 30)
