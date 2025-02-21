# 라이브러리 
import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam

# 1. 데이터 준비
(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
# 정규화
x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0
# 카테고리화
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 2. 모델 작성(Sequential)
cnn = Sequential()
cnn.add(Conv2D(6,(5,5), padding='same',activation='relu', input_shape=(28,28,1)))
cnn.add(MaxPooling2D(pool_size=(2,2),strides=2))
cnn.add(Conv2D(16,(5,5), padding='valid',activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),strides=2))
cnn.add(Conv2D(120,(5,5), padding='valid',activation='relu'))
cnn.add(Flatten())
cnn.add(Dense(units=84, activation='relu'))
cnn.add(Dense(units=10, activation='softmax'))

# 3. 모델 컴파일
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
# cnn.summary()
# 4. 모델 훈련
cnn.fit(x_train,y_train, batch_size=128, epochs=30, validation_data=(x_test,y_test), verbose=2)

# 5. 훈련 검증
res = cnn.evaluate(x_test, y_test, verbose=0)
print('정확률=', res[1]*100)