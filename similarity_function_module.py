from sklearn.svm import OneClassSVM,NuSVC,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pickle
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
import matplotlib.pyplot as plt
from numpy.random import seed
seed(7)
from tensorflow import set_random_seed
set_random_seed(11)

def load_datasets():
    dataset={}
    target ={}
    for i in range(3):

        objects = []
        filename= "datos/piezas_task_%i"%i
        with (open(filename, "rb")) as openfile:

            while True:

                try:
                    objects.append(pickle.load(openfile))
                except EOFError:

                    break

        dataset[i] = objects
    return dataset
def confusion_matrix_l():
    clasi = "datos/SMV_class_"

    clasificadores = []



    #
    test = [[] for _ in range(3)]



    for i in range(3):

        f1 = open(clasi+"%i"%i, "r+b")

        with open("datos/X_test_%i"%i,"r+b") as openfile:
            while True:
                try:
                    test[i].append(pickle.load(openfile))
                except EOFError:
                    break
        clasificadores.append(pickle.load(f1))
        f1.close()



    for i in range(3):
        for j in range(3):

            clas = clasificadores[j]
            x=test[i]
            y=clas.predict(x)
            print(f"Clase {i:d} Clasificador {j:d}")
            pos = len(np.where(y==1)[0])
            neg =len(y)-pos
            print(f"Positivos: {pos:d}   Negativos:{neg:d}")


def flatten(X):
    '''
    Flatten a 3D array.

    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.

    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return (flattened_X)

def scale(X, scaler):
    '''
    Scale 3D array.

    Inputs
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    scaler       A scaler object, e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize

    Output
    X            Scaled 3D array.
    '''
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])

    return X
def create_LTSautoencoder(task):
    data = load_datasets()
    X_0 = data[task][:25000]
    n = len(X_0[0])
    time_step =int(n/27)
    for i,x in enumerate(X_0):
        X_0[i] = preprocessing.scale(x.reshape(time_step,27))
    X_0 = np.array(X_0)
    X_0 = shuffle(X_0)
    X_0_train,X_test = train_test_split(X_0,test_size=0.1)
    X_0_train, X_0_val = train_test_split(X_0_train,test_size=0.2)

    layers = [  LSTM(64,activation="relu",input_shape=(time_step,27),return_sequences=True),
                keras.layers.LSTM(32,activation="relu",return_sequences=True),
                keras.layers.LSTM(16,activation="relu",return_sequences=False),
                RepeatVector(time_step),
                keras.layers.LSTM(16, activation="relu", return_sequences=True),
                keras.layers.BatchNormalization(),
                keras.layers.LSTM(32, activation="relu", return_sequences=True),
                LSTM(64, activation="relu", return_sequences=True),
                TimeDistributed(Dense(27))]
    model= keras.Sequential(layers)
    model.compile(loss = "mse",optimizer=optimizers.Adam(0.0001))
    with tf.device("GPU:0"):
        history = model.fit(X_0_train,X_0_train,batch_size=128,epochs=200,verbose=2,validation_data=(X_0_val,X_0_val))

    plt.plot(history.history['loss'], linewidth=2, label='Train')
    plt.plot(history.history['val_loss'], linewidth=2, label='Valid')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig("datos/error.png")
    plt.figure(figsize=(10,10))



    X_1 = data[task+1][:100]
    for i,x in enumerate(X_1):
        X_1[i] = preprocessing.scale(x.reshape(time_step,27))
    X_1 = np.array(X_1)
    X_1 =shuffle(X_1)

    prueba_1 = model.predict(X_1)
    prueba_0 =model.predict(X_test[:100])
    mse_1 = np.mean(np.power(flatten(X_1) - flatten(prueba_1), 2), axis=1)
    mse_0 = np.mean(np.power(flatten(X_test[:100]) - flatten(prueba_0), 2), axis=1)

    X = np.vstack((X_0_train,X_1))
    lo =[0]*18000
    lo.extend([1]*X_1.shape[0])
    y = np.array(lo)



    # ax.hlines(0.1, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    plt.scatter(range(100),mse_1)
    plt.scatter(range(100),mse_0,)
    plt.title("Reconstruction error for different classes")
    plt.legend([f"Anomalias (Tarea {task+1:d})",f"No anomalias (Tarea {task:d})"])
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    plt.savefig("datos/clasificacion.png")

    co = f"datos/LTSM_AUTOENCODER_20000_task_{task:d}.h5"
    model.save(co)

    prueba_1 = model.predict(X)
    mse_1 = np.mean(np.power(flatten(X) - flatten(prueba_1), 2), axis=1)
    plt.figure(  figsize=(10,10))
    precision_rt, recall_rt, threshold_rt = precision_recall_curve(y, mse_1)
    plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
    plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
    plt.title('Precision and recall for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend()
    plt.show()

    return model



def test_autoencoder(task):
    N=1000
    data = load_datasets()
    X_1 = shuffle(data[task+1])[:N]
    time_step = int(len(X_1[0])/27)
    for i,x in enumerate(X_1):
        X_1[i] = preprocessing.scale(x.reshape(time_step,27))
    # X_1 = np.array(X_1)

    y1 = [1]*N

    X_0 = shuffle(data[task])[:N]
    for i,x in enumerate(X_1):
        X_0[i] = preprocessing.scale(x.reshape(time_step,27))
    # X_0 = np.array(X_0)


    model= keras.models.load_model(f"datos/LTSM_AUTOENCODER_20000_task_{task:d}.h5")



    X_1.extend(X_0)
    y0 = [0]*N
    y1.extend(y0)
    y1=np.array(y1)
    X_1 = np.array(X_1)
    prueba_1 = model.predict(X_1)
    mse_1 = np.mean(np.power(flatten(X_1) - flatten(prueba_1), 2), axis=1)
    precision_rt, recall_rt, threshold_rt = precision_recall_curve(y1,mse_1)
    plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
    plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
    plt.title('Precision and recall for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend()
    plt.show()
    TRESHOLD=0.02









if __name__ == '__main__':
    model=lstm_autoencoder = create_LTSautoencoder(0)

    # test_autoencoder(1)


    # data=load_datasets()
    # clasificadores = []
    #
    #
    # for i in range(3):
    #     for j in range(i+1,3):
    #         X = data[i]
    #         Xj = data [j]
    #         y = [i]*len(X)
    #         yj = [j]*len(Xj)
    #         y.extend(yj)
    #         X.extend(Xj)
    #         X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
    #
    #         X_train,y_train = shuffle(X_train,y_train)
    #         X_test,y_test = shuffle(X_test,y_test)
    #
    #         X_train = preprocessing.scale(X_train)
    #         X_test = preprocessing.scale(X_test)
    #
    #
    #
    #         cos=SVC()
    #         cos.fit(X_train[:5000,:],y_train[:5000])
    #         co = "datos/SMV_class_%i_%i"%(i,j )
    #         with open(co,'w+b') as fp:
    #             pickle.dump(cos , fp)
    #
    #         cos2 = LogisticRegression()
    #         cos2.fit(X_train,y_train)
    #         co = f"datos/Logistic_class_{i:d}_{j:d}"
    #         with open(co, 'w+b') as fp:
    #             pickle.dump(cos2, fp)
    #
    #
    #         # print("Ya entrene para la clase %i"%i)
    #
    #         filenameX = "datos/X_test_%i_%i"%(i,j)
    #     # for x in X_test:
    #         with open(filenameX, 'a+b') as fp:
    #             pickle.dump(X_test, fp)
    #         filenameY = "datos/y_test_%i_%i" % (i, j)
    #         with open(filenameY, 'a+b') as fp:
    #             pickle.dump(y_test, fp)
    #
    #         print(f"Matriz de confunción de las clases {i:d} y {j:d} del SVM")
    #         con =confusion_matrix(y_test[:5000],cos.predict(X_test[:5000,:]))
    #         print(con)
    #         print(f"Matriz de confución de las clases {i:d} y {j:d} de la Logistic regression")
    #         con = confusion_matrix(y_test, cos2.predict(X_test))
    #         print(con)

    # filename = "datos/clasificadores"
    #
    # with open(filename, 'a+b') as fp:
    #     pickle.dump(clasificadores, fp)
    # confusion_matrix()
