from rtnn import Layers as l
from rtnn import network as net
from rtnn import Visualizer as vis
from rtnn import inputUtil as iu
import irisInputSignal as inp

import numpy as np

from timeit import default_timer as timer

startTimer = timer()

epochs = 1
# Time interval
t0 = 0
T = 1600

inputSize = 8
outputSize = 3

print('Timesteps per example: T=', T)

network = net.Network()
inputLayer = l.InputLayer(inputSize)
network.addLayer(inputLayer)
hiddenLayer = l.DuoLateralInhibitoryLayer(6)
network.addLayer(hiddenLayer)
outputLayer = l.DuoLateralInhibitoryLayer(outputSize)
network.addLayer(outputLayer)

adjustments = []
differences = []

trainingHistory = []

startSimulation = timer()
print('Time to build the network: ', startSimulation - startTimer, ' seconds')

for epoch in range(epochs):
    startEpoch = timer()
    print('Epoch number ', epoch)
    for exampleId in range(0, len(inp.y_train)):
        print('exampleId: ', exampleId)
        preneuralExcitoryInputsSeries = inp.makeTrainInputCurrent(exampleId, T)
        postneuralExcitoryInputsSeries = inp.makeOutputCurrent(exampleId)# duration 1700

        timeBiases = iu.makeTimeBiases(outputLayer,
                                       excitatoryBiasesT=postneuralExcitoryInputsSeries,
                                       inhibitoryBiasesT=iu.noSpike(len(postneuralExcitoryInputsSeries),
                                                                    len(outputLayer)))

        hiddenNoise = iu.poissonDistributedSpikeTrain(T, len(hiddenLayer), [0.008, 0.1, 0.2, 0.15, 0.2, 0.18])
        timeBiases = iu.makeTimeBiases(hiddenLayer,
                                       excitatoryBiasesT=hiddenNoise,
                                       inhibitoryBiasesT=iu.noSpike(len(hiddenNoise),
                                                                    len(hiddenLayer)),
                                       timeBiases=timeBiases)

        # Run Simulation
        for time in range(t0, T):
            preneuralInhibitoryInputs = iu.noSpike(T, 8)[time]
            preneuralExcitoryInputs = preneuralExcitoryInputsSeries[time]

            layerBiasDict = timeBiases[time]
            network.step(preneuralExcitoryInputs, preneuralInhibitoryInputs,
                         layerBiasDict=layerBiasDict, ignorePreneurons=False)
            if exampleId % 10 == 0:
                #print('Collect Training Data for every 10th example.')
                trainingHistory.append(network.logNetwork()) #Process finished with exit code 137 (interrupted by signal 9: SIGKILL)

    if exampleId % 10 == 0:
        print('Network Adjustment.')
        #network.adjust()
    endEpoch = timer()
    print('Epoch ', epoch, ' ran in ', endEpoch - startEpoch, ' seconds')

testHistory = []

for testId in range(0, len(inp.y_test)):
    print('testId: ', testId)
    preneuralExcitoryInputsSeries = inp.makeTestInputCurrent(testId, T)
    timeBiases = {}
    # Run Simulation
    for time in range(t0, T):
        preneuralInhibitoryInputs = iu.noSpike(T, 8)[time]
        preneuralExcitoryInputs = preneuralExcitoryInputsSeries[time]
        network.step(preneuralExcitoryInputs, preneuralInhibitoryInputs)
        testHistory.append(network.logNetwork())

startPlotting = timer()
print('Simulation time: ', startPlotting - startSimulation, ' seconds')

trainingHistory = np.array(trainingHistory)
vis.visualizeNetwork(trainingHistory, visualizeSynapses=False)

testHistory = np.array(testHistory)
vis.visualizeNetwork(testHistory, visualizeSynapses=False)

endTimer = timer()
print('Plotting time: ', endTimer - startPlotting, ' seconds')
print('Elapsed time: ', endTimer - startTimer, ' seconds')
print('done')

#4800.43 seconds
