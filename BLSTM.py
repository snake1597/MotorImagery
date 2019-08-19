import numpy as np
from loadData import load
import matplotlib.pyplot as plt
import csv, os, sys, copy, pywt, time
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from loadDataBci_VI import loadBciVI
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

OUTPUT_SIZE = 4
#path = './data/old_data/avgdata/myself'
path = './data/testdata'
#subject = 'S192T_LRTEST S192T_BKBK S192T_TO'.split(' ')
subject = 'ray'.split(' ')
getDataClass = load(subNames=subject, path=path, notEqual='n',isRand=True)
(x_train, y_train,name) = getDataClass.loadTrainDataFromTxt()
# loadData = loadBciVI(subNames=subject, path=path)
# (x_train, y_train) = loadData.loadTrainDataFromTxt()

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, ReduceLROnPlateau
from keras import initializers,regularizers
from keras.models import load_model
from keras.utils import plot_model
# from keras import regularizers
from keras.layers.normalization import BatchNormalization
# from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
np.random.seed(1337)  # for reproducibility
# data pre-processing
#Y_train = np_utils.to_categorical(Y_train, num_classes=OUTPUT_SIZE)
#y_test = np_utils.to_categorical(y_test, num_classes=OUTPUT_SIZE)

ntest  = int(x_train.shape[0] * 0.1)     # number of testing data
ntrain = x_train.shape[0] - ntest        # number of training data
x_test  = x_train[ntrain:]
y_test  = y_train[ntrain:]
x_train = x_train[:ntrain]
y_train = y_train[:ntrain]
print('X_train shape:', x_train.shape)
print('Y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
y_train = np_utils.to_categorical(y_train, num_classes=OUTPUT_SIZE)

BATCH_SIZE = 64
TIME_STEPS = x_train.shape[1]
INPUT_SIZE = x_train.shape[2]
LR = 0.005 # 0.005 0.0005
first_layer = 12
second_layer = 60
#rmsprop = RMSprop(lr=LR, decay=0.0002)
#adam = Adam(lr=LR, decay=0.0003, clipvalue=0.5)
#sgd = SGD(lr=LR, momentum=0.0, decay=0.0, nesterov=False)
opt = RMSprop(lr=LR,decay=0.0002)

print('Build model...')
tStart = time.time()
model = Sequential()
### BI-LSTM ###
model.add(Bidirectional(LSTM(first_layer,return_sequences=True,activation='tanh'),input_shape=x_train.shape[1:],merge_mode='sum'))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(second_layer,return_sequences=False,activation='tanh'),merge_mode='sum'))
model.add(Dropout(0.5))

# model.add(Dense(second_layer, activation='tanh'))
# model.add(Dropout(0.5))
### LSTM ###
# model.add(LSTM(first_layer, input_shape=x_train.shape[1:],return_sequences=False,))
# #model.add(LSTM(second_layer,return_sequences=False,))
# model.add(Dropout(0.2))

model.add(Dense(OUTPUT_SIZE, activation='softmax'))
model.summary()
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,   
    metrics=['accuracy']
)
DateTime = str(time.strftime("%m%d_%H%M", time.localtime()))
# checkpoint path
filepath="./weights_improvement/Bidirectional_LSTM/bci_{0}_{1}_{2}_{3}".format(DateTime, subject, first_layer, second_layer)
filepath=filepath+'_{val_acc:.2f}.hdf5'

saveBestModel = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
#saveBestModel = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
earlystop  = EarlyStopping(monitor='val_loss', patience=40, verbose=1, mode='auto')

callbacks_list = [saveBestModel]

print('Training ------------')
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=100,
    verbose=2,
    validation_split=0.1,
    shuffle=True,
    callbacks=callbacks_list,
)

# score = model.evaluate(x_test, y_test, verbose=0)
# print ('Test loss:', score[0])
# print ('Test accuracy:', score[1])
tEnd = time.time()
print ("It cost %f sec" % (tEnd - tStart))  
del model  # deletes the existing model
mypath = './weights_improvement/Bidirectional_LSTM/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
fileName = mypath + onlyfiles[-1]
print ('load model: ',fileName)
model = load_model(fileName)

index_shuf  = list(zip(x_test, y_test))
index_shuf = sorted(index_shuf, key=lambda x:x[1])
x_test, y_test= zip(*index_shuf)
y_test = np_utils.to_categorical(y_test, num_classes=OUTPUT_SIZE)
x_test = np.array(x_test,dtype=float)
y_test = np.array(y_test,dtype=int)

score = model.evaluate(x_test, y_test, verbose=0)
print ('Test loss:', score[0])
print ('Test accuracy:', score[1])

def matchArray(arrayOne, arrayTwo):
    successCount = 0
    indexOne = arrayOne.argmax(axis=1)
    indexTwo = arrayTwo.argmax(axis=1)
    for i in range(0,indexOne.shape[0]):
        rowMax = np.amax(arrayOne[i])
        #print (rowMax, ':', indexOne[i] , '-', indexTwo[i])
        if indexOne[i] == indexTwo[i] and rowMax >= 0.8 :
        #if indexOne[i] == indexTwo[i]:
            successCount+=1
    accuracy = successCount / float(indexOne.shape[0])
    return accuracy

resultLabel = model.predict(x_test)
print ('test acc: %0.2f' % (matchArray(resultLabel, y_test)))
for i in range(len(y_test)):
    print ('test predict:', resultLabel[i]) #[:10]
    print ('test label: ', y_test[i])

# list all data in history
print(history.history.keys())
plt.figure(1)
#plt.figure(figsize=(2, 1))
# summarize history for accuracy
plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.subplots_adjust(hspace=0.5)
nowDateTime = str(time.strftime("%Y%m%d_%H%M", time.localtime()))
_imgPath = "./compilerResault/training_{1}_{0}_lose.png".format(nowDateTime,subject)
plt.savefig(_imgPath, dpi=720)
print("save Plot Image %s" % (_imgPath))
plt.show()
plt.close()

print("end")