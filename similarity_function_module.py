from sklearn.svm import OneClassSVM,NuSVC,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np
from sklearn import preprocessing
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









if __name__ == '__main__':
    data=load_datasets()
    clasificadores = []


    for i in range(3):
        for j in range(i+1,3):
            X = data[i]
            Xj = data [j]
            y = [i]*len(X)
            yj = [j]*len(Xj)
            y.extend(yj)
            X.extend(Xj)
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

            X_train,y_train = shuffle(X_train,y_train)
            X_test,y_test = shuffle(X_test,y_test)

            X_train = preprocessing.scale(X_train)
            X_test = preprocessing.scale(X_test)



            cos=SVC()
            cos.fit(X_train[:5000,:],y_train[:5000])
            co = "datos/SMV_class_%i_%i"%(i,j )
            with open(co,'w+b') as fp:
                pickle.dump(cos , fp)

            cos2 = LogisticRegression()
            cos2.fit(X_train,y_train)
            co = f"datos/Logistic_class_{i:d}_{j:d}"
            with open(co, 'w+b') as fp:
                pickle.dump(cos2, fp)


            # print("Ya entrene para la clase %i"%i)

            filenameX = "datos/X_test_%i_%i"%(i,j)
        # for x in X_test:
            with open(filenameX, 'a+b') as fp:
                pickle.dump(X_test, fp)
            filenameY = "datos/y_test_%i_%i" % (i, j)
            with open(filenameY, 'a+b') as fp:
                pickle.dump(y_test, fp)

            print(f"Matriz de confunción de las clases {i:d} y {j:d} del SVM")
            con =confusion_matrix(y_test[:5000],cos.predict(X_test[:5000,:]))
            print(con)
            print(f"Matriz de confución de las clases {i:d} y {j:d} de la Logistic regression")
            con = confusion_matrix(y_test, cos2.predict(X_test))
            print(con)

    # filename = "datos/clasificadores"
    #
    # with open(filename, 'a+b') as fp:
    #     pickle.dump(clasificadores, fp)
    # confusion_matrix()
