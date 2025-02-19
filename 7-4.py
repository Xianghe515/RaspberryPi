# MNIST를 이용한 딥러닝 학습 및 결과 코드 
import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 학습 데이터 로드
(x_train,y_train),(x_test,y_test)=ds.mnist.load_data()
# 데이터 값에 대한 처리... 
# 784 값은 앞서 얻었던 (60000, 28, 28)의 값을 변형한 것임.
# 2차원 텐서의 값을 1차원 형식으로 표현.  
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
# 각 값을 0 ~ 1 사이의 값으로 변환
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
# 각 레이블을 표현하는 값을 categorical로 변환
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)
print(y_train[0]) # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

# 모델(딥러닝, 활성화=relu)
dmlp=Sequential()
dmlp.add(Dense(units=1024,activation='relu',input_shape=(784,)))
dmlp.add(Dense(units=512,activation='relu'))
dmlp.add(Dense(units=512,activation='relu'))
dmlp.add(Dense(units=10,activation='softmax'))

# 모델 컴파일
dmlp.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)
# 모델 학습
hist = dmlp.fit(x_train,y_train,
                batch_size=128,
                epochs=50,
                validation_data=(x_test,y_test),
                verbose=2)
# 모델 저장
dmlp.save('dmlp_trained.h5')
import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.grid()
plt.show()

