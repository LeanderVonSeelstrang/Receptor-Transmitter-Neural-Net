import numpy as np

from rtnn import Layers as l
from rtnn import network as net
from rtnn import Visualizer as vis
from rtnn import inputUtil as iu

from timeit import default_timer as timer

startTimer = timer()


inputSize = 4
outputSize = 3
T = 700

networkHistory = []

print('Simulating until T=', T)

network = net.Network()
inputLayer = l.InputLayer(inputSize)
network.addLayer(inputLayer)

# outputLayer = l.DuoLateralInhibitoryLayer(outputSize)
# network.addLayer(outputLayer)

outputLayer = l.ExcitatoryLateralInhibitoryLayer(outputSize)
network.addLayer(outputLayer)



#min p for T = 1500: 0.008
#max p for T = 1500: 0.5

timeBiases = iu.makeTimeBiases(outputLayer,
                               excitatoryBiasesT=iu.poissonDistributedSpikeTrain(T, len(outputLayer), [0.006, 0.0055, 0.005]),
                               inhibitoryBiasesT=iu.noSpike(T+2, len(outputLayer)))

#timeBiases = iu.makeTimeBiases(outputLayer, iu.biasTrain)
#timeBiases = iu.makeTimeBiases(hiddenLayer, iu.biasTrain, timeBiases)

startSimulation = timer()
print('Time to build the network: ', startSimulation - startTimer, ' seconds')

for t in range(T):
    excitatoryIn = iu.tetanus0breaktetanus1[t] #iu.poissonDistributedSpikeTrain(T, len(inputLayer), 0.05)[t] #iu.noSpike(2, inputSize)[0] #iu.epilepsie[t] #
    inhibitoryIn = iu.noSpike(2, inputSize)[0]
    layerBiasDict = {} #timeBiases[t]
    network.step(excitatoryIn, inhibitoryIn, layerBiasDict=layerBiasDict) #, ignorePreneurons=True)
    networkHistory.append(network.logNetwork())
    if t % 1000 == 0:
        print('t: ', t)
      #  print('\ninputLayer:\n', inputLayer)
      #  print('\nhiddenLayer:\n', hiddenLayer)
        print('\noutputLayer:\n', outputLayer)

startPlotting = timer()
print('Simulation time: ', startPlotting - startSimulation, ' seconds')

networkHistory = np.array(networkHistory)
vis.visualizeNetwork(networkHistory, visualizeSynapses=True)

#outputLayer = network.layers[-1]
#outputneurons = outputLayer.postneurons

endTimer = timer()
print('Plotting time: ', endTimer - startPlotting, ' seconds')
print('Elapsed time: ', endTimer - startTimer, ' seconds')
print('done')
