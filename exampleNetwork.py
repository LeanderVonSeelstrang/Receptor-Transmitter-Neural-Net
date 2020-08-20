import numpy as np

from rtnn import Layers as l
from rtnn import network as net
from rtnn import Visualizer as vis
from rtnn import inputUtil as iu

from timeit import default_timer as timer

startTimer = timer()


inputSize = 4
outputSize = 3
T = 1600 - 900

networkHistory = []

print('Simulating until T=', T)

network = net.Network()
inputLayer = l.InputLayer(inputSize)
network.addLayer(inputLayer)
hiddenLayer = l.LateralInhibitoryLayer(5)
network.addLayer(hiddenLayer)
hiddenLayer2 = l.LateralInhibitoryLayer(5)
network.addLayer(hiddenLayer2)
hiddenLayer3 = l.LateralInhibitoryLayer(5)
network.addLayer(hiddenLayer3)
outputLayer = l.LateralInhibitoryLayer(outputSize)
network.addLayer(outputLayer)


timeBiases = iu.makeTimeBiases(outputLayer,
                               excitatoryBiasesT=iu.poissonDistributedSpikeTrain(T, len(outputLayer), [0.6, 0.55, 0.5]),
                               inhibitoryBiasesT=iu.noSpike(T+2, len(outputLayer)))
timeBiases = iu.makeTimeBiases(hiddenLayer,
                               excitatoryBiasesT=iu.poissonDistributedSpikeTrain(T, len(hiddenLayer),
                                                                                 [0.005, 0.1, 0.2, 0.3, 0.4]),
                               inhibitoryBiasesT=iu.noSpike(T, len(hiddenLayer)), timeBiases=timeBiases)
timeBiases = iu.makeTimeBiases(hiddenLayer2,
                               excitatoryBiasesT=iu.poissonDistributedSpikeTrain(T, len(hiddenLayer2),
                                                                                 [0.002, 0.004, 0.005, 0.008, 0.01]),
                               inhibitoryBiasesT=iu.noSpike(T, len(hiddenLayer2)), timeBiases=timeBiases)
timeBiases = iu.makeTimeBiases(hiddenLayer3,
                               excitatoryBiasesT=iu.poissonDistributedSpikeTrain(T, len(hiddenLayer3),
                                                                                 [0.6, 0.55, 0.5, 0.45, 0.4]),
                               inhibitoryBiasesT=iu.noSpike(T, len(hiddenLayer3)), timeBiases=timeBiases)

#timeBiases = iu.makeTimeBiases(outputLayer, iu.biasTrain)
#timeBiases = iu.makeTimeBiases(hiddenLayer, iu.biasTrain, timeBiases)

startSimulation = timer()
print('Time to build the network: ', startSimulation - startTimer, ' seconds')

for t in range(T):
    excitatoryIn = iu.tetanus0breaktetanus1[t] #iu.poissonDistributedSpikeTrain(T, len(inputLayer), 0.05)[t] #iu.noSpike(2, inputSize)[0] #iu.epilepsie[t] #
    inhibitoryIn = iu.noSpike(2, inputSize)[0]
    layerBiasDict = timeBiases[t]
    network.step(excitatoryIn, inhibitoryIn, layerBiasDict=layerBiasDict, ignorePreneurons=True)
    networkHistory.append(network.logNetwork())
    if t % 1000 == 0:
        print('t: ', t)
      #  print('\ninputLayer:\n', inputLayer)
      #  print('\nhiddenLayer:\n', hiddenLayer)
        print('\noutputLayer:\n', outputLayer)

startPlotting = timer()
print('Simulation time: ', startPlotting - startSimulation, ' seconds')

networkHistory = np.array(networkHistory)
vis.visualizeNetwork(networkHistory, visualizeSynapses=False)

#outputLayer = network.layers[-1]
#outputneurons = outputLayer.postneurons

endTimer = timer()
print('Plotting time: ', endTimer - startPlotting, ' seconds')
print('Elapsed time: ', endTimer - startTimer, ' seconds')
print('done')
