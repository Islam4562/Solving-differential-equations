import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Генерация данных для симуляции краш-теста
# Колонки: скорость, сила удара, деформация, повреждения (0 - нет, 1 - поврежден)
np.random.seed(42)

n_samples = 1000
speed = np.random.uniform(10, 100, n_samples)  # скорость автомобиля (м/с)
impact_force = np.random.uniform(5000, 50000, n_samples)  # сила удара (Н)
deformation = np.random.uniform(0.1, 1.0, n_samples)  # деформация (м)
damage = (speed * deformation * impact_force > 1000000).astype(int)  # Примерная зависимость повреждений от параметров

# Создание DataFrame с данными
data = pd.DataFrame({
    'speed': speed,
    'impact_force': impact_force,
    'deformation': deformation,
    'damage': damage
})

# Разделение на признаки и целевую переменную
X = data[['speed', 'impact_force', 'deformation']]
y = data['damage']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели случайного леса
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Прогнозирование на тестовых данных
y_pred = model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Прогнозирование повреждений для новых данных
new_data = np.array([[60, 25000, 0.5]])  # Пример: скорость 60 м/с, сила удара 25000 Н, деформация 0.5 м
predicted_damage = model.predict(new_data)
print(f'Predicted damage: {predicted_damage[0]}')  # 0 - нет повреждений, 1 - поврежден
