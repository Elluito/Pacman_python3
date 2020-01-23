import numpy as np
import os
import sys
import glob
import matplotlib.pyplot as plt


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
    # parser.add_option('-l', '--layout', dest='layout',
    #                   help=default('the LAYOUT_FILE from which to load the map layout'),
    #                   metavar='LAYOUT_FILE', default='mediumClassic')
    # parser.add_option('-p', '--pacman', dest='pacman',
    #                   help=default('the agent TYPE in the pacmanAgents module to use'),
    #                   metavar='TYPE', default='KeyboardAgent')


    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = dict()
    args["file"]=options.file

    # if args['layout'] == None: raise Exception("The layout " + options.layout + " cannot be found")

    # Choose a Pacman agent
    # noKeyboard = options.gameToReplay == None and (options.textGraphics or options.quietGraphics)
    #
    # if options.numTraining > 0:
    #     args['numTraining'] = options.numTraining

    #
    #
    # args['pacman'] = pacman
    #
    #
    # # Choose a ghost agent
    #
    # args['ghosts'] = [ghostType(i + 1) for i in range(options.numGhosts)]

    # Choose a display format
    # if options.quietGraphics:
    #     import textDisplay
    #     args['display'] = textDisplay.NullGraphics()
    # elif options.textGraphics:
    #     import textDisplay
    #     textDisplay.SLEEP_TIME = options.frameTime
    #     args['display'] = textDisplay.PacmanGraphics()
    # else:
    #     import graphicsDisplay

        ##### aquí es cuando pinta las cosas que debe pintar

    #     args['display'] = graphicsDisplay.PacmanGraphics(options.zoom, frameTime=options.frameTime)
    # args['numGames'] = options.numGames
    # args['record'] = options.record
    # args['catchExceptions'] = options.catchExceptions
    # args['timeout'] = options.timeout
    # args["difficulty"] = options.difficulty
    # args["inicio"] = options.inicio
    # args["final"] = options.final
    #
    # # Special case: recorded games don't use the runGames method or args structure
    # if options.gameToReplay != None:
    #     print('Replaying recorded game %s.' % options.gameToReplay)
    #     import cPickle
    #     f = open(options.gameToReplay)
    #     try:
    #         recorded = cPickle.load(f)
    #     finally:
    #         f.close()
    #     recorded['display'] = args['display']
    #     replayGame(**recorded)
    #     sys.exit(0)

    return args


def graficar_todos_juntos(max_number):
    names= ["Tarea_0","Tarea_1","Tarea_2","Tarea_3"]
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
    print(all_datos)
    for i,prom in enumerate(all_datos):
        x = np.linspace(0, len(prom) * 10, len(prom))
        plt.plot(x, prom,label=names[i])
    plt.xlabel("Episodios transcurridos")
    plt.ylabel("Probabilidad  de ganar")
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

    graficar_todos_juntos(2)