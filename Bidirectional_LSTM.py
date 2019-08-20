import numpy as np
from loadData import load
import matplotlib.pyplot as plt
import os, copy, pywt, time
from os import listdir
from os.path import isfile, join
import tensorflow as tf

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import RMSprop
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import load_model
from keras.utils import plot_model

np.random.seed(1337)  # for reproducibility
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# load train dataset
OUTPUT_SIZE = 2
path = './bci2003'
subject = 'trainData_DWT'.split(' ')
getDataClass = load(subNames=subject, path=path, notEqual='n',isRand=True)
(x_train, y_train) = getDataClass.loadTrainDataFromTxt()
y_train = np_utils.to_categorical(y_train, num_classes=OUTPUT_SIZE)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
# load test dataset
path = './bci2003'
subject = 'testData_DWT'.split(' ')
getDataClass = load(subNames=subject, path=path, notEqual='n',isRand=True)
(x_test, y_test) = getDataClass.loadTrainDataFromTxt()
print('x_test shape:', x_train.shape)
print('y_test shape:', y_train.shape)

BATCH_SIZE = 16
EPOCHS = 200
TIME_STEPS = x_train.shape[1]
INPUT_SIZE = x_train.shape[2]
LR = 0.005 
n_hidden = 10
opt = RMSprop(lr=LR,decay=0.0002)

# Build model BiLSTM
print('Build model...')
tStart = time.time()
model = Sequential()
model.add(Bidirectional(LSTM(n_hidden,return_sequences=False,activation='tanh'),input_shape=x_train.shape[1:],merge_mode='sum'))
model.add(Dropout(0.2))
model.add(Dense(OUTPUT_SIZE, activation='softmax'))
model.summary()
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,   
    metrics=['accuracy']
)

DateTime = str(time.strftime("%m%d_%H%M", time.localtime()))
# checkpoint path
directory = './modelSave'
if not os.path.exists(directory):
    os.makedirs(directory)
filepath="{0}/bci_{1}_{2}_{3}".format(directory, DateTime, subject, n_hidden)
filepath=filepath+'_{val_acc:.2f}.hdf5'
saveBestModel = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [saveBestModel]

print('Training ------------')
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    validation_split=0.1,
    shuffle=True,
    callbacks=callbacks_list,
)
tEnd = time.time()
print ("Training cost %f sec" % (tEnd - tStart))  

del model  # deletes the existing model
mypath = '{0}/'.format(directory)
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
fileName = mypath + onlyfiles[-1]
print ('load model: ',fileName)
model = load_model(fileName)
# sort the test data
index_shuf  = list(zip(x_test, y_test))
index_shuf = sorted(index_shuf, key=lambda x:x[1])
x_test, y_test= zip(*index_shuf)
y_test = np_utils.to_categorical(y_test, num_classes=OUTPUT_SIZE)
x_test = np.array(x_test,dtype=float)
y_test = np.array(y_test,dtype=int)
# testing model
score = model.evaluate(x_test, y_test, verbose=0)
print ('Test loss:', score[0])
print ('Test accuracy:', score[1])

# threshold check
def matchArray(arrayOne, arrayTwo):
    successCount = 0
    indexOne = arrayOne.argmax(axis=1)
    indexTwo = arrayTwo.argmax(axis=1)
    for i in range(0,indexOne.shape[0]):
        rowMax = np.amax(arrayOne[i])
        if indexOne[i] == indexTwo[i] and rowMax >= 0.8 :
            successCount+=1
    accuracy = successCount / float(indexOne.shape[0])
    return accuracy

resultLabel = model.predict(x_test)
print ('pass threshold accuracy: %0.2f' % (matchArray(resultLabel, y_test)))
# show testing result
# for i in range(len(y_test)):
#     print ('test predict:', resultLabel[i])
#     print ('test label: ', y_test[i])

# list all data in history
print(history.history.keys())
plt.figure(1)
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
# save loss image
# nowDateTime = str(time.strftime("%Y%m%d_%H%M", time.localtime()))
# _imgPath = "./training_{1}_{0}_loss.png".format(nowDateTime,subject)
# plt.savefig(_imgPath, dpi=720)
# print("save Plot Image %s" % (_imgPath))
plt.show()
#plt.close()
print("end")