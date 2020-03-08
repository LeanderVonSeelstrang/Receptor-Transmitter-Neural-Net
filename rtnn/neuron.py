import numpy as np

from . import synapse as s # the two classes are technically independent, but I did not want to write one function twice

SPIKING_THRESHHOLD = 150
ACTION_POTENTIAL_VALUE = 600

# Initialize Neuron.
def initNeuron(exitoryInCellCount0, inhibitoryInCellCount0, remainingRefractoryTime0, spiking0, numberOfSpikes0,
               timeSinceLastSpike0, recentActivity0, highlyActiveTime0, noActionTime0):
    neuronLearningIndicators = np.array([numberOfSpikes0, timeSinceLastSpike0,
                                         recentActivity0, highlyActiveTime0, noActionTime0])
    return np.array([exitoryInCellCount0, inhibitoryInCellCount0, remainingRefractoryTime0,
                     spiking0, neuronLearningIndicators])

class Neuron:
    def __init__(self, exitoryInCellCount0 = 0, inhibitoryInCellCount0 = 0, remainingRefractoryTime0 = 0,
                spiking0 = False, numberOfSpikes0 = 0, timeSinceLastSpike0 = -1,
                recentActivity0 = 0.0, highlyActiveTime0 = 0, noActionTime0 = 0):
        self.neuron = initNeuron(exitoryInCellCount0, inhibitoryInCellCount0, remainingRefractoryTime0, spiking0,
                                 numberOfSpikes0, timeSinceLastSpike0,
                                 recentActivity0, highlyActiveTime0, noActionTime0)
        self.isSpiking = self.neuron[3]
        self.neuronParameters = sampleNeuronParameters(expectedLeak=2, varianceLeak=1,
                                                       expectedMaximalInhibition=SPIKING_THRESHHOLD*2, #randomly chosen
                                                       varianceMaximalInhibition=2, expectedInhibitionLeak=2,
                                                       varianceInhibitionLeak=2, spikingThreshold=SPIKING_THRESHHOLD,
                                                       actionPotentialValue=ACTION_POTENTIAL_VALUE, refractoryTime=9)

    def step(self,newExitoryInputPotential, newInhibitoryInputPotential):
        self.neuron = exitoryInhibitoryNeuronDynamics(newExitoryInputPotential, newInhibitoryInputPotential,
                                                      self.neuron, self.neuronParameters)
        self.isSpiking = self.neuron[3]
        return self

    def updateNeuronParameters(self, spikingThreshold, actionPotentialValue, leak,
                               refractoryTime, maximalInhibition, inhibitionLeak):
        return np.array(
            [spikingThreshold, actionPotentialValue, leak, refractoryTime, maximalInhibition, inhibitionLeak])

    def __str__(self):
        self.excitatoryInCell = self.neuron[0]
        self.inhibitoryInCell = self.neuron[1]
        self.remainingRefractoryTime = self.neuron[2]
        self.isSpiking = self.neuron[3]
        self.neuronLearningIndicators = self.neuron[4]

        self.numberOfSpikes = self.neuronLearningIndicators[0]
        self.highlyActiveTime = self.neuronLearningIndicators[3]
        self.noActionTime = self.neuronLearningIndicators[4]

        return ('Exitatory potential: ' + str(self.excitatoryInCell) + '\n' +
                'Inhibitory potential: ' + str(self.inhibitoryInCell) + '\n' +
                'Remaining refractory time: ' + str(self.remainingRefractoryTime) + '\n' +
                'Is currently spiking: ' + str(self.isSpiking) + '\n\n' +
                'Number of spikes: ' + str(self.numberOfSpikes) + '\n' +
                'Highly active time: ' + str(self.highlyActiveTime) + '\n' +
                'Inactive time: ' + str(self.noActionTime) + '\n\n')

    def detailedDeciption(self):
        self.excitatoryInCell = self.neuron[0]
        self.inhibitoryInCell = self.neuron[1]
        self.remainingRefractoryTime = self.neuron[2]
        self.isSpiking = self.neuron[3]
        self.neuronLearningIndicators = self.neuron[4]

        self.numberOfSpikes = self.neuronLearningIndicators[0]
        self.timeSinceLastSpike = self.neuronLearningIndicators[1]
        self.recentActivity = self.neuronLearningIndicators[2]
        self.highlyActiveTime = self.neuronLearningIndicators[3]
        self.noActionTime = self.neuronLearningIndicators[4]

        self.spikingThreshold = self.neuronParameters[0]
        self.actionPotentialValue = self.neuronParameters[1]
        self.neuronLeak = self.neuronParameters[2]
        self.refractoryTime = self.neuronParameters[3]
        self.maximalInhibition = self.neuronParameters[4]
        self.inhibitionNeuronLeak = self.neuronParameters[5]

        return ('Exitory potential: ' + str(self.excitatoryInCell) + '\n' +
                'Inhibitory potential: ' + str(self.inhibitoryInCell) + '\n' +
                'Remaining refractory time: ' + str(self.remainingRefractoryTime) + '\n' +
                'Is currently spiking: ' + str(self.isSpiking) + '\n\n' +
                'Learning Indicators: ' + '\n' +
                'Number of spikes: ' + str(self.numberOfSpikes) + '\n' +
                'Time since last spike: ' + str(self.timeSinceLastSpike) + '\n' +
                'Recent Activity: ' + str(self.recentActivity) + '\n' +
                'Highly active time: ' + str(self.highlyActiveTime) + '\n' +
                'Inactive time: ' + str(self.noActionTime) + '\n\n' +
                'Neuron parameters: \n' +
                'Spiking threshold: ' + str(self.spikingThreshold) + '\n' +
                'Action potential value: ' + str(self.actionPotentialValue) + '\n' +
                'Neuron leak: ' + str(self.neuronLeak) + '\n' +
                'Refractory time: ' + str(self.refractoryTime) + '\n' +
                'Maximal inhibition: ' + str(self.maximalInhibition) + '\n' +
                'Inhibition leak: ' + str(self.inhibitionNeuronLeak) + '\n\n\n')







def exitoryInhibitoryNeuronDynamics(newExitoryInputPotential, newInhibitoryInputPotential, neuron, neuronParameters):

    '''
     the spike value should be reasonable high, so it is not accidently crossed by the inputPotential.
     A too high spike value does not cause harm – does not exist
     A too low spike value might ignore a spike value, when the inputPotential + newInputPotential
     crosses not only the threshold,
     but also the spike value, then this spike is ignored.
    :param newExitoryInputPotential:
    :param newInhibitoryInputPotential:
    :param neuron:
    :param neuronParameters:
    :return: the next state, after the new input potential was computed
    '''

    ignoreAllInputsDuringRefractoryTime = True
    exitoryPotential, inhibitoryPotential, remainingRefractoryTime, spiking, neuronLearningIndicators = neuron
    numberOfSpikes, timeSinceLastSpike, recentActivity, highlyActiveTime, noActionTime = neuronLearningIndicators
    threshold, spikeValue, leak, refractoryTime, maximalInhibition, inhibitionLeak = neuronParameters

    recentActivity = s.releasePercentageDecay(recentActivity, defaultReleasePercentage=0.1, decaySpeed=0.008)
    neuronLearningIndicators[2] = recentActivity

    if recentActivity > 0.65: # more or less randomly chosen hyperparameter
        highlyActiveTime = highlyActiveTime + 1
        neuronLearningIndicators[3] = highlyActiveTime

    if recentActivity < 0.12: # more or less randomly chosen hyperparameter
        noActionTime = noActionTime + 1
        neuronLearningIndicators[4] = noActionTime

    # calculate the new inhibitory potential within the neuron
    if (inhibitoryPotential - inhibitionLeak) + newInhibitoryInputPotential <= maximalInhibition:
        inhibitoryPotential = (inhibitoryPotential - inhibitionLeak) + newInhibitoryInputPotential
        if inhibitoryPotential <= 0:
            inhibitoryPotential = 0
    else:
        inhibitoryPotential = maximalInhibition
    neuron[1] = inhibitoryPotential
    # calculate the new exitory Potential within the neuron
    exitoryPotential = exitoryPotential + newExitoryInputPotential
    # include both inhibitory and exitory potential to calculate the new input potential
    inputPotential = (exitoryPotential - leak) - inhibitoryPotential

    if spiking:  # second method to detect if the neuron just spiked. Maybe this should be moved up.
        neuronLearningIndicators[1] = timeSinceLastSpike + 1
        neuron[0] = 0
        neuron[1] = newInhibitoryInputPotential
        neuron[2] = refractoryTime
        neuron[3] = False
        return neuron

    # calculating next remaining refractory time
    if remainingRefractoryTime - 1 >= 0:
       # remainingRefractoryTime = remainingRefractoryTime - 1
        neuron[2] = remainingRefractoryTime - 1
        if ignoreAllInputsDuringRefractoryTime:
            neuronLearningIndicators[1] = timeSinceLastSpike + 1
            neuron[1] = newInhibitoryInputPotential
            neuron[3] = False
            return neuron
    else:
        #remainingRefractoryTime = 0
        neuron[2] = 0

    if inputPotential < 0:  # the input potential should never be negative, if it is inhibitory Potential is left over
        if (exitoryPotential - leak) >= 0:
            inhibitoryPotential = inhibitoryPotential - (exitoryPotential - leak)
        neuron[1] = inhibitoryPotential
        neuronLearningIndicators[1] = timeSinceLastSpike + 1
        return neuron # no spike, no potential

    if inputPotential > threshold:  # induce a spike
        recentActivity = s.releasePercentageIncrease(recentActivity, maximumReleasePercentage=1, growthSpeed=0.9)
        neuronLearningIndicators[0] = numberOfSpikes + 1
        neuronLearningIndicators[1] = 0
        neuronLearningIndicators[2] = recentActivity

        neuron[0] = spikeValue
        neuron[1] = 0
        neuron[2] = refractoryTime
        neuron[3] = True
        return neuron

    else:  # 'normal' behaviour - not spiking, not after a spike
        neuronLearningIndicators[1] = timeSinceLastSpike + 1
        if inputPotential - leak <= 0: # more negative then positive potential
            if (exitoryPotential - leak) >= 0:
                inhibitoryPotential = inhibitoryPotential - (exitoryPotential - leak)
                if inhibitoryPotential < 0:
                    inhibitoryPotential = 0
            neuron[0] = 0
            neuron[1] = inhibitoryPotential
            return neuron
        else: # more excitatory potential
            neuron[0] = (inputPotential - leak)
            neuron[1] = 0
            return neuron


# -------------------------Define Neuron Parameters------------------------------
# actionPotentialValue
# spikingThreshold
# refractoryTime

# samples a list of parameters for one neuron
# spikingThreshold, actionPotentialValue and refractoryTime are assumed to be global constants
def sampleNeuronParameters(expectedLeak = 2, varianceLeak = 1, expectedMaximalInhibition = 1000,
                           varianceMaximalInhibition=50, expectedInhibitionLeak = 2, varianceInhibitionLeak = 2,
                           spikingThreshold = SPIKING_THRESHHOLD, actionPotentialValue = ACTION_POTENTIAL_VALUE,
                           refractoryTime = 9):
    spikingThreshold = spikingThreshold
    actionPotentialValue = actionPotentialValue
    leak = np.floor(np.absolute(np.random.logistic(expectedLeak, varianceLeak)))
    refractoryTime = refractoryTime
    maximalInhibition = np.floor(np.absolute(np.random.logistic(expectedMaximalInhibition, varianceMaximalInhibition)))
    inhibitionLeak = np.ceil(np.absolute(np.random.logistic(expectedInhibitionLeak, varianceInhibitionLeak)))
    return np.array([spikingThreshold, actionPotentialValue, leak, refractoryTime, maximalInhibition, inhibitionLeak])
