from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
import numpy as np

# Загружает DataSet одежды.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Преобразуем картинки в плоский вектор.
x_train = x_train.reshape(60_000, 784)
# Нормализуем данные: до в НС поступали бы числа от 0 до 255, после: от 0 до 1.
# Нужно для алгоритмов оптимизации обучения НС.
x_train = x_train / 255

# Преобразуем метки в категории.
y_train = utils.to_categorical(y_train, 10)

classes = ['T-shirt/top', 'Trouser', 'Pullover',
           'Dress', 'Coat', 'Sandal', 'Shirt',
           'Sneaker', 'Bag', 'Ankle boot']

# Создаем последовательную модель.
model = Sequential()

# Добавляем уровни нейронов.
model.add(Dense(800, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Компилируем модель.
# p.s. функция ошибки "categorical_crossentropy" хорошо подходит
#      бля задач классификаций, если классов больше двух.
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
print(model.summary())

# Рисуем НС и сохраняем в файл.
utils.plot_model(model, to_file='plot_model.png', show_shapes=True, show_layer_names=False)

# Обучаем НС.
model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=1)

# Финальный тест.
predictions = model.predict(x_train)
print("Результаты по 1-ой картинке:", predictions[0])
print("Наш ответ:       ", classes[np.argmax(predictions[0])])
print("Правильный ответ:", classes[np.argmax(y_train[0])])
