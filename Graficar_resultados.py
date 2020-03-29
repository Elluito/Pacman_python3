import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import stats
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


def plot_mean_and_CI(t,mean, lb, ub, color_mean=None, color_shading=None,label=""):
    # plot the shaded range of the confidence intervals
    plt.fill_between(t, ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(t,mean, c=color_mean,label=label)


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

def dar_up_lw_bound(y,window,variance=False):
    i=0
    up=[]
    lw=[]
    t = stats.t.ppf(1 - 0.025, window - 1)
    while i+window<=len(y):
        pedazo = y[i:i+window]
        std=np.std(pedazo)
        if not variance:
            up.append(-(std/np.sqrt(window))*t)
            lw.append(+(std/np.sqrt(window))*t)
        else:
            up.append(-(std) )
            lw.append(std)

        i+=1
    return np.array(up),np.array(lw)

def graficar_todos_juntos(max_number,window):
    names= ["T2-exponencial-prob","T0-T2-prob"]
    # names=["T1-exponencial-score","T0-T1-exponencial-score"]
    # names=["Phi","Epsilon"]
    all_datos = []
    all_std=[]

    for i in range(max_number+1):
        # directory=os.chdir(os.path.abspath(__file__))
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
        os.chdir("..")
        if len(datos.shape)>=2:
            prom = np.mean(datos, axis=1)
            std = np.std(datos,axis=1)
        else:
            prom=datos
        t = stats.t.ppf(1 - 0.025,datos.shape[1]-1)
        N=len( glob.glob("*.txt"))
        all_datos.append(prom)
        # all_std.append((std/np.sqrt(N))*t)
        all_std.append(std)
    # print(all_datos)
    for i,prom in enumerate(all_datos):
        y=running_mean(window,prom)
        # std_mean=running_mean(window,all_std[i])

        up,lw=dar_up_lw_bound(prom,window,variance=True)
        x = np.linspace(0, len(prom), len(up))
        plot_mean_and_CI(x,y,y+lw,y+up,color_mean="C{}".format(i),color_shading="C{}".format(i),label=names[i].replace("-exponencial-prob","").replace("-prob","").replace("-exponencial-score","").replace("-","->"))
        # plt.plot(x,y,label=names[i])

    plt.xlabel("Episodes",fontsize=20)
    s=""
    if "score" in names[i]:
        s="Final score of episode"
    else :
        s="Winning probability "
    plt.ylabel(s,fontsize=20)
    plt.title("Estimate with running mean of size {}".format(window),fontsize=20)
    plt.legend(prop={"size":15})
    fig = plt.gcf()
    ax=fig.axes[0]
    ax.tick_params(labelsize=15)

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




def visualize_trail():

    trail=[]
    f="datos/history"
    fp=open(f,"r+b")
    p=True
    while p:

        try:
            k=pickle.load(fp)
            print(k)
            trail.append(k)
        except :
            p=False


    mapa= np.zeros((18,18))
    mapa[10,8]=float("Inf")
    for lista in trail:
        for tupla in lista:
            # if tupla[0]!=10 or tupla[1] !=10:
                mapa[tupla[0],tupla[1]]+=1

    l=plt.matshow(np.rot90(mapa))


    # plt.xticks([1, 0], [f"Anomalies (Task {task + 1:d})", f"No anomalies (Task {task:d})"])
    # plt.yticks([0.6, -0.4], [f"Anomalies (Task {task + 1:d})", f"No anomalies (Task {task:d})"], rotation=90)

    # plt.xlabel("Predicted class", fontsize=20)
    # plt.ylabel("True Class", fontsize=20)
    # plt.title("Task {}".format(task))
    #
    # ax.text(0,0,ma[0,0])
    # ax.text(0,1,ma[0,1])
    # ax.text(1,0,ma[1,0])
    # ax.text(1,1,ma[1,1])
    plt.title("Map",fontsize=20)
    fig = plt.gcf()
    ax = fig.axes[0]
    cbar = fig.colorbar(l)
    # ax.text(0, 0, "{0:.2f}".format(ma[0, 0]), ha="center", va="center", color="k", fontsize=20)
    # ax.text(0, 1, "{0:.2f}".format(ma[0, 1]), ha="center", va="center", color="k", fontsize=20)
    # ax.text(1, 0, "{0:.2f}".format(ma[1, 0]), ha="center", va="center", color="k", fontsize=20)
    # ax.text(1, 1, "{0:.2f}".format(ma[1, 1]), ha="center", va="center", color="k", fontsize=20)
    ax.tick_params(labelsize=20)
    fig.set_size_inches(10, 10)
    plt.show()
    # plt.savefig("datos/mapa_trail.png")










if __name__ == '__main__':

    graficar_todos_juntos(1,500)
    # visualize_trail()