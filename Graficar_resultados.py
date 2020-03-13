import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def default(str):
    return str + ' [Default: %default]'

def readCommand(argv):
    """
    Processes the command used to run pacman from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python pacman.py <options>
    EXAMPLES:   (1) python pacman.py
                    - starts an interactive game
                (2) python pacman.py --layout smallClassic --zoom 2
                OR  python pacman.py -l smallClassic -z 2
                    - starts an interactive game on a smaller board, zoomed in
    """
    parser = OptionParser(usageStr)

    parser.add_option('-c', '--carpeta', dest='file', type='str',
                      help=default("Where file is located"), default=os.path.dirname(os.path.abspath(__file__))+"/Tarea_0")

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = dict()
    args["file"]=options.file


    return args


def graficar_todos_juntos(max_number,window):
    names= ["T2-exponencial-prob","T1-T2-exponencial-prob"]
    # names=["T2-exponencial-score","T1-T2-exponencial-score"]
    names=["Phi","Epsilon"]
    all_datos = []
    for i in range(max_number+1):
        directory=(os.path.dirname(os.path.abspath(__file__)))+"\\"+names[i]
        os.chdir(directory)
        datos=None
        for file in glob.glob("*.txt"):
            print(file)
            if datos is None:
                datos = np.loadtxt(str(directory) + "\\" + str(file))
            else:
                # rows,columns = tuple(datos.shape)
                nuevo = np.loadtxt(str(directory) + "\\" + str(file))
                datos = np.column_stack((datos, nuevo))
        if len(datos.shape)>=2:
            prom = np.mean(datos, axis=1)
        else:
            prom=datos
        all_datos.append(prom)
    # print(all_datos)
    for i,prom in enumerate(all_datos):
        y=running_mean(window,prom)
        x = np.linspace(0, len(prom) , len(y))
        plt.plot(x,y,label=names[i])

    plt.xlabel("Episodes")
    s=""
    if "score" in names[i]:
        s="Final score of episode"
    else :
        s="Winning probability "
    plt.ylabel(s)
    plt.title("Estimate with running mean of size {}".format(window))
    plt.legend()

    plt.show()

def running_mean(N,x):

    return np.convolve(x, np.ones((N,)) / N, mode='valid')

def graficar_uno():
    args = readCommand(sys.argv[1:])
    directory = None
    if len(args["file"]) < 10:
        directory = str(os.path.dirname(os.path.abspath(__file__)))
        directory = directory + str(args["file"])

    else:
        directory = args["file"]

    # upload all  filee in "directory"
    datos = None
    os.chdir(directory)
    for file in glob.glob("*.txt"):
        if datos is None:
            datos = np.loadtxt(str(directory) + "\\" + str(file))
        else:
            # rows,columns = tuple(datos.shape)
            nuevo = np.loadtxt(str(directory) + "\\" + str(file))
            datos = np.column_stack((datos, nuevo))

    print(datos)
    prom = np.mean(datos, axis=1)
    x = np.linspace(0, len(prom) * 10, len(prom))
    plt.plot(x, prom)
    plt.xlabel("Episodes")

    plt.ylabel("winning probabolity")
    plt.legend([args["file"][1:]])

    nombre_fig = "prob" + "_Tarea_0"

    # plt.savefig("datos\\"+nombre_fig+".png")

    plt.show()

















if __name__ == '__main__':

    graficar_todos_juntos(1,1)