from rtnn import Layers as L
from rtnn import network as net
from rtnn import Visualizer as Vis
from rtnn import inputUtil as Iu
import numpy as np


inputSize = 1
outputSize = 1
t0 = 0
T = 1501

calculations = 600

networkHistory = []

network = net.Network()
inputLayer = L.InputLayer(inputSize)
network.addLayer(inputLayer)
outputLayer = L.DuoLateralInhibitoryLayer(outputSize, excitatoryTransmitterExpectation=5200, excitatoryTransmitterVariance=0,
                                          excitatoryReceptorExpectation=1600, excitatoryReceptorVariance=0,
                                          inhibitoryTransmitterExpectation=3500, inhibitoryTransmitterVariance=0,
                                          inhibitoryReceptorExpectation=1000, inhibitoryReceptorVariance=0)
network.addLayer(outputLayer)

for difference in range(int(-(calculations / 2)), int(calculations / 2)):
    print('difference: ', difference)
    preneuralExcitoryInputsSeries, postneuralExcitoryInputsSeries = Iu.spikeTimeDifference(difference)
    postneuralInhibitoryInputsSeries = Iu.noSpike(len(postneuralExcitoryInputsSeries), outputSize) # to ignore the inhibitory potentials
    timeBiases = Iu.makeTimeBiases(outputLayer, postneuralExcitoryInputsSeries, postneuralInhibitoryInputsSeries)
    print('Simulating until T=', T)
    # Run Simulation
    for time in range(t0, T):
        excitatoryIn = preneuralExcitoryInputsSeries[time]
        # iu.poissonDistributedSpikeTrain(T, len(inputLayer), 0.05)[t] #iu.noSpike(2, inputSize)[0] #iu.epilepsie[t] #
        inhibitoryIn = Iu.noSpike(2, inputSize)[0]
        layerBiasDict = timeBiases[time]
        network.step(excitatoryIn, inhibitoryIn, layerBiasDict=layerBiasDict, ignorePreneurons=True)
        networkHistory.append(network.logNetwork())
        if time % 500 == 0:
            print('t: ', time)
        # print('\ninputLayer:\n', inputLayer)
        # print('\nhiddenLayer:\n', hiddenLayer)
        # print('\noutputLayer:\n', outputLayer)

networkHistory = np.array(networkHistory)
Vis.visualiseLearningSignalHistory(networkHistory, calculations, T)
#Vis.visualizeNetwork(networkHistory, visualizeSynapses=False)
