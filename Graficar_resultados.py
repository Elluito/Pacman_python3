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


def graficar_todos_juntos(max_number):
    names= ["T2 exponencial","T1-T2 lineal"]
    all_datos = []
    for i in range(max_number+1):
        directory=(os.path.dirname(os.path.abspath(__file__)))+"\\"+names[i]
        os.chdir(directory)
        datos=None
        for file in glob.glob("*.txt"):
            if datos is None:
                datos = np.loadtxt(str(directory) + "\\" + str(file))
            else:
                # rows,columns = tuple(datos.shape)
                nuevo = np.loadtxt(str(directory) + "\\" + str(file))
                datos = np.column_stack((datos, nuevo))
        prom = np.mean(datos, axis=1)
        all_datos.append(prom)
    # print(all_datos)
    for i,prom in enumerate(all_datos):
        x = np.linspace(0, len(prom) * 10, len(prom))
        plt.plot(x,  smooth(prom,0),label=names[i])
    plt.xlabel("Episodes")
    plt.ylabel("Winning probability")
    plt.legend()

    plt.show()



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
    plt.xlabel("Episodios transcurridos")
    plt.ylabel("Probabilidad  de ganar")
    plt.legend([args["file"][1:]])

    nombre_fig = "prob" + "_Tarea_0"

    # plt.savefig("datos\\"+nombre_fig+".png")

    plt.show()

















if __name__ == '__main__':

    graficar_todos_juntos(1)