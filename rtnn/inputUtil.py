import numpy as np

from . import neuron as n

def spikeNeuron(neuronToSpike, noInputNeurons=4, thresholdValue = n.SPIKING_THRESHHOLD):
    spikePotential = np.zeros(noInputNeurons)
    spikePotential[neuronToSpike] = thresholdValue + 30
    return np.array([spikePotential])


def noSpike(duration, noInputNeurons=4):
    return np.zeros(noInputNeurons * duration).reshape(duration, noInputNeurons)


def addLayerBiasesToBiasDict(layer, excitatoryBias, inhibitoryBias, biasDict = None):

    if not biasDict:
        biasDict = {}
    biasDict[layer] = (excitatoryBias, inhibitoryBias)
    return biasDict


def makeTimeBiases(layer, excitatoryBiasesT = None, inhibitoryBiasesT = None, timeBiases = None):
    if (not (excitatoryBiasesT is None) and not (inhibitoryBiasesT is None)):
        if len(excitatoryBiasesT) != len(inhibitoryBiasesT):
            raise ValueError('excitatoryBiasesT and inhibitoryBiasesT must have the same length, but have the lengths '
                             + str(len(excitatoryBiasesT)) + ' and ' + str(len(inhibitoryBiasesT)) + '.')
    if not timeBiases:
        timeBiases = {}
    for time in range(len(excitatoryBiasesT)):
        if not (excitatoryBiasesT is None):
            excitatoryBias = excitatoryBiasesT[time]
        else:
            excitatoryBias = None
        if not (inhibitoryBiasesT is None):
            inhibitoryBias = inhibitoryBiasesT[time]
        else:
            inhibitoryBias = None

        if time in timeBiases:
            biasDict = timeBiases[time]
            timeBiases[time] = addLayerBiasesToBiasDict(layer, excitatoryBias, inhibitoryBias, biasDict)
        else:
            timeBiases[time] = addLayerBiasesToBiasDict(layer, excitatoryBias, inhibitoryBias)

    return timeBiases

def poissonDistributedNeuronStep(probabilityOfFiring, timeStep = 1):
    probabilityOfSurviveOneStep = 1 - probabilityOfFiring * timeStep
    uniform = np.random.random_sample()
    if uniform - probabilityOfSurviveOneStep > 0:
        return True
    else:
        return False

def poissonDistributedSpikeTrain(duration, numberNeuronsInLayer = 5, firingProbabilities = [0.01,0.01,0.01,0.01,0.01]):
    spikings = []
    if len(firingProbabilities) != numberNeuronsInLayer:
        raise ValueError('The number of neurons does not align with the number of firingProbabilities. ')
    while duration > 0:
        for neuronIx in range(numberNeuronsInLayer):
            probabilityOfFiring = firingProbabilities[neuronIx]
            if poissonDistributedNeuronStep(probabilityOfFiring):
                spikings.append(spikeNeuron(neuronIx, numberNeuronsInLayer))
                #print('spike neuron: ', neuronIx)
                duration -= 1
            else:
                spikings.append(noSpike(1, numberNeuronsInLayer))
                duration -= 1
    spikings = np.concatenate(spikings, axis=0)
    return spikings



spikeNeuron0 = spikeNeuron(0)
spikeNeuron1 = spikeNeuron(1)
spikeNeuron2 = spikeNeuron(2)
spikeNeuron3 = spikeNeuron(3)

smallTetanus0breaktetanus1 = np.concatenate(
    [noSpike(300), spikeNeuron0, noSpike(20), spikeNeuron0, noSpike(20), spikeNeuron0, noSpike(20), spikeNeuron0,
     noSpike(20), spikeNeuron0, noSpike(20), spikeNeuron0, noSpike(20), spikeNeuron0, noSpike(20), spikeNeuron0,
     noSpike(20), spikeNeuron0, noSpike(20), spikeNeuron0, noSpike(300), spikeNeuron1, noSpike(20), spikeNeuron1,
     noSpike(20), spikeNeuron1, noSpike(20), spikeNeuron1, noSpike(20), spikeNeuron1, noSpike(20), spikeNeuron1,
     noSpike(20), spikeNeuron1, noSpike(20), spikeNeuron1, noSpike(20), spikeNeuron1, noSpike(20), spikeNeuron1,
     noSpike(20), noSpike(300), spikeNeuron2, noSpike(10), spikeNeuron3, noSpike(10), spikeNeuron2, noSpike(10),
     spikeNeuron3, noSpike(20), spikeNeuron2, noSpike(10), spikeNeuron3, noSpike(10), spikeNeuron2, noSpike(10),
     spikeNeuron3, noSpike(10), spikeNeuron2, noSpike(10), spikeNeuron3, noSpike(10), spikeNeuron2, noSpike(10),
     spikeNeuron3, noSpike(10), spikeNeuron2, noSpike(10), spikeNeuron3, noSpike(10), spikeNeuron2, noSpike(10),
     spikeNeuron3, noSpike(10), spikeNeuron2, noSpike(10), spikeNeuron3, noSpike(10)], axis=0)
tetanus0breaktetanus1 = np.tile(smallTetanus0breaktetanus1, (4, 1))

#print('poissonDistributedSpikeTrain:', poissonDistributedSpikeTrain(200))
#print('tetanus0breaktetanus1:', tetanus0breaktetanus1)

smallEpilepsie = np.concatenate([noSpike(50), spikeNeuron0, noSpike(5), spikeNeuron1, noSpike(5), spikeNeuron2,
                                 noSpike(5), spikeNeuron3, noSpike(5), spikeNeuron0, noSpike(5), spikeNeuron1,
                                 noSpike(5), spikeNeuron2, noSpike(5), spikeNeuron3, noSpike(5), spikeNeuron0,
                                 noSpike(5), spikeNeuron1, noSpike(5), spikeNeuron2, noSpike(5), spikeNeuron3,
                                 noSpike(5), spikeNeuron0, noSpike(5), spikeNeuron1, noSpike(5), spikeNeuron2,
                                 noSpike(5), spikeNeuron3, noSpike(5), spikeNeuron0, noSpike(5), spikeNeuron1,
                                 noSpike(5), spikeNeuron2, noSpike(5), spikeNeuron3, noSpike(5), spikeNeuron0,
                                 noSpike(5), spikeNeuron1, noSpike(5), spikeNeuron2, noSpike(5), spikeNeuron3,
                                 noSpike(5), spikeNeuron0, noSpike(5), spikeNeuron1, noSpike(5), spikeNeuron2,
                                 noSpike(5), spikeNeuron3, noSpike(5), spikeNeuron0, noSpike(5), spikeNeuron1,
                                 noSpike(5), spikeNeuron2, noSpike(5), spikeNeuron3, noSpike(5), spikeNeuron0,
                                 noSpike(5), spikeNeuron1, noSpike(5), spikeNeuron2, noSpike(5), spikeNeuron3,
                                 noSpike(5), spikeNeuron0, noSpike(5), spikeNeuron1, noSpike(5), spikeNeuron2,
                                 noSpike(5), spikeNeuron3, noSpike(5), spikeNeuron0, noSpike(5), spikeNeuron1,
                                 noSpike(5), spikeNeuron2, noSpike(5), spikeNeuron3, noSpike(5)])

epilepsie = np.tile(smallEpilepsie, (8, 1))

biasSpike2 = spikeNeuron(2,3)
smallBiasTrain = np.concatenate([noSpike(150,3), biasSpike2, noSpike(20,3), biasSpike2, noSpike(20,3), biasSpike2,
                                 noSpike(20,3), biasSpike2, noSpike(20,3), biasSpike2, noSpike(20,3), biasSpike2,
                                 noSpike(20,3), biasSpike2, noSpike(20,3), biasSpike2, noSpike(20,3), biasSpike2,
                                 noSpike(20,3), biasSpike2, noSpike(20,3), biasSpike2, noSpike(20,3), biasSpike2,
                                 noSpike(20,3), biasSpike2, noSpike(20,3), noSpike(20,3)], axis=0)
biasTrain = np.tile(smallBiasTrain, (7, 1))

'''
timeBiases = iu.makeTimeBiases(outputLayer, iu.biasTrain)
timeBiases = iu.makeTimeBiases(hiddenLayer, iu.biasTrain, timeBiases)
'''


def spikeTimeDifference(timeDifference, numberNeurons=1):
    '''
    This function is used to generate spike trains for different timeDifferences.
    It will be used to visualize the stdp effect in the end.
    :param timeDifference: between preneuronal and postneuronal spike
    :return: a spike train for the preNeuron and one for the postneuron
    '''
    preneuronSpikeTrain = np.tile(np.concatenate([noSpike(700, numberNeurons), spikeNeuron(0, numberNeurons),
                                                  noSpike(10, numberNeurons), spikeNeuron(0, numberNeurons),
                                                  noSpike(10, numberNeurons), spikeNeuron(0, numberNeurons),
                                                  noSpike(10, numberNeurons), spikeNeuron(0, numberNeurons),
                                                  noSpike(10, numberNeurons), spikeNeuron(0, numberNeurons),
                                                  noSpike(600, 1)]), (4, 1))
    postneuronSpikeTrain = np.tile(np.concatenate([noSpike(700 + timeDifference, numberNeurons),
                                                   spikeNeuron(0, numberNeurons), noSpike(10, numberNeurons),
                                                   spikeNeuron(0, numberNeurons), noSpike(10, numberNeurons),
                                                   spikeNeuron(0, numberNeurons), noSpike(10, numberNeurons),
                                                   spikeNeuron(0, numberNeurons), noSpike(10, numberNeurons),
                                                   spikeNeuron(0, numberNeurons), noSpike(10, numberNeurons),
                                                   noSpike(600,1)]), (4, 1))
    return preneuronSpikeTrain, postneuronSpikeTrain
