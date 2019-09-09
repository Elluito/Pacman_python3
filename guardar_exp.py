# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 18:29:26 2019

@author: edwin.torres
"""

import time
import os

# Inicio del experimento
t0 = time.time()
iniTime = time.strftime("%H:%M:%S")
iniDate = time.strftime("%d/%m/%Y")

# Capturar directorio de trabajo
wd_path = os.getcwd()

# Capturar fecha y hora
# Capturando la hora diferenciamos los expermientos
file_date = time.strftime("%Y%m%d-%H%M%S")    # no deberia llamarse file_date mejor dir_date

# Crear ruta directorio agregando parametros generales que identifiquen el experimento
# se agrega el file_date al niombre del directorio
final_directory = os.path.join(wd_path,'prueba_' + data_a + '_' + data_b + '_'+file_date)

# Crear directorio
if not os.path.exists(final_directory):
        os.makedirs(final_directory)


'''

script del experimento

cualquier grafica o resultado debe guardarse
en el directorio previamente creado: final_directory

'''

# Final del experimento
# serializar el model a JSON
model_json = model.to_json()

# Crear y guardar archivo JSON en el directorio
with open(final_directory+'\\'+ 'prueba_xx' +'_'+'tipo_de_red'+'_Q_NNet.json', "w") as json_file:
    json_file.write(model_json)


endTime = time.strftime("%H:%M:%S")
endDate = time.strftime("%d/%m/%Y")

total_time = (time.time()-t0)/60

# Estos parametros aplican para mis experimentos
# crear archivo que guarde TODOS los parametros de configuracion del experimento
resumenParametros = open(final_directory+'\\'+ transf +'_'+'paramsRL_' +parametro+'_'+d_level+'_'+file_date+'.txt', "w")
resumenParametros.write(('Transfer:\t%s' %(TRANSFER) + '\n' +
                         'Transfer from:\t%s' %(name_old) + '\n' +
                         'Tran. rate phi:\t%1s' %(phi) + '\n' +
                         'White strategy:\t' +white.__name__+'_'+d_level+ '\n' +
                         'Experiments:\t%d' % (numtests) + '\n' +
                         'Trials:\t\t%d' % (nTrials) + '\n' +
                         'Games:\t\t%d' % (nGames) + '\n' +
                         'Iterations:\t%d' % (nI) + '\n' +
                         'gamma:\t\t%1s' % (gamma) + '\n' +
                         'epsilon:\t%1s' % (epsilon) + '\n' +
                         'FinalEpsilon:\t%1s' % (finalEpsilon) + '\n' +
                         'Neurons hidden:\t%d' % (nHidden_01) + '\n' +
                         'reward win:\t%d' % (REWARD_WIN) + '\n' +
                         'reward loss:\t%d' % (REWARD_LOSS) + '\n' +
                         'reward otw:\t%d' % (REWARD_OTW) + '\n' +
                         'Initial time:\t' + iniTime + ' ' +iniDate + '\n' +
                         'Final time:\t' + endTime + ' ' +endDate + '\n' +
                         'Tiempo total:\t%f min' % total_time + '\n'))#.encode('utf8', 'replace'))
resumenParametros.close()

