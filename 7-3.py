# SGD와 ADAM의 비교 시각화 코드 
import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD

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

# 모델(SGD)
mlp_sgd=Sequential()
mlp_sgd.add(Dense(units=512,activation='tanh',input_shape=(784,)))
mlp_sgd.add(Dense(units=10,activation='softmax'))

# 모델 컴파일(SGD)
mlp_sgd.compile(loss='MSE',optimizer=SGD(learning_rate=0.01),metrics=['accuracy'])
# 모델 학습(SGD)
hist_sgd = mlp_sgd.fit(x_train,y_train,batch_size=128,epochs=20,validation_data=(x_test,y_test),verbose=2)
# 모델 검증(SGD)
res=mlp_sgd.evaluate(x_test,y_test,verbose=0)
print('(SGD)정확률=',res[1]*100)

# 모델(Adam)
mlp_adam=Sequential()
mlp_adam.add(Dense(units=512,activation='tanh',input_shape=(784,)))
mlp_adam.add(Dense(units=10,activation='softmax'))

# 모델 컴파일(Adam)
mlp_adam.compile(loss='MSE',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
# 모델 학습(Adam)
hist_adam = mlp_adam.fit(x_train,y_train,batch_size=128,epochs=20,validation_data=(x_test,y_test),verbose=2)

# 모델 검증(Adam)
res=mlp_adam.evaluate(x_test,y_test,verbose=0)
print('정확률=',res[1]*100)


import matplotlib.pyplot as plt

plt.plot(hist_sgd.history['accuracy'],'r--')
plt.plot(hist_sgd.history['val_accuracy'],'r')
plt.plot(hist_adam.history['accuracy'],'b--')
plt.plot(hist_adam.history['val_accuracy'],'b')
plt.title('Comparison of SGD and Adam optimizers')
plt.ylim((0.7,1.0))
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train_sgd','val_sgd','train_adam','val_adam'])
plt.grid()
plt.show