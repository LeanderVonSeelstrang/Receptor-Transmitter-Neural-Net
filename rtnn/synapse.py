import numpy as np
import random

from . import hyperparameters as hyperparameters


MAX_CLIFT_TRANSMITTER_COUNT = hyperparameters.MAX_CLIFT_TRANSMITTER_COUNT #60000 #500000 #130000

COLLISION_SPEED = hyperparameters.COLLISION_SPEED #0.00001386294361119890618834464242916353136151
RECEPTOR_BUILD_SPEED = hyperparameters.RECEPTOR_BUILD_SPEED #0.00897309
TRANSMITTER_BUILD_SPEED = hyperparameters.TRANSMITTER_BUILD_SPEED #0.000002351386294361119890618834464242916353136151
EXPONENTIAL_CLIFT_LEAK_SPEED = hyperparameters.EXPONENTIAL_CLIFT_LEAK_SPEED #0.004

TIME_DIFFERENCE_SENSITIVITY_LEARNING_SIGNAL = hyperparameters.TIME_DIFFERENCE_SENSITIVITY_LEARNING_SIGNAL #0.00005138629

RELEASE_PERCENTAGE = hyperparameters.RELEASE_PERCENTAGE #0.1

LINEAR_CLIFT_LEAK = hyperparameters.LINEAR_CLIFT_LEAK #3


# basically irrelevant - just to ensure definition
# see Layers.py 
EXPECTED_MAXIMAL_RECEPTOR_COUNT = 1200
VARIANCE_MAXIMAL_RECEPTOR_COUNTI = 103

EXPECTED_MAX_TRANSMITTERCOUNT = 3500
VARIANCE_MAX_TRANSMITTERCOUNT = 101



def getState(syn):
    return syn.synapse


def getAcceptedTransmitters(syn):
    return syn.synapse[3]


def resetAcceptedTransmitters(syn):
    syn.synapse[3] = 0
    return syn

#def getHyperparameters(maxOfMaxVesicles, maxOfMaxReceptors, minOfMaxVesicles, minOfMaxReceptors):
 #   return np.array([maxOfMaxVesicles, maxOfMaxReceptors, minOfMaxVesicles, minOfMaxReceptors])


# Initialize a synapse. Holds the current state of the synapse.
def initSynapse(transVetrikel0, transGap0, rezeptor0, transPost0, releasePercentage0, substanceA0, substanceB0,
                substanceC0, accumulatedLeak):
    learningIndicators = np.array([substanceA0, substanceB0, substanceC0, accumulatedLeak])
    return np.array([transVetrikel0, transGap0, rezeptor0, transPost0, releasePercentage0, learningIndicators])


class Synapse:

    '''
    A synapse transmitts information from one neuron to another neuron via chemical transmitter substances.

    Different parameters can be modulated:
    A preneuron can decide how many transmitters are released into the synaptic clift.
    A postneuron can decide how much information is accepted by modulating the number of corresponding receptors.
    '''

    def __init__(self, expectedMaximalTransmitterCount0=EXPECTED_MAX_TRANSMITTERCOUNT,
                 varianceMaximalTransmitterCount0=VARIANCE_MAX_TRANSMITTERCOUNT,
                 expectedMaximalRezeptorCount0=EXPECTED_MAXIMAL_RECEPTOR_COUNT,
                 varianceMaximalRezeptorCount0=VARIANCE_MAXIMAL_RECEPTOR_COUNTI,
                 linearCliftLeak=LINEAR_CLIFT_LEAK,
                 defaultReleasePercentage=None,
                 releasePercentage0=RELEASE_PERCENTAGE,
                 transVetrikel0=1, transGap0=0, rezeptor0=1, transPost0=0,
                 substanceA0=0, substanceB0=0, substanceC0=0, accumulatedLeak=0):
        self.synapse = initSynapse(transVetrikel0, transGap0, rezeptor0, transPost0, releasePercentage0,
                                   substanceA0, substanceB0, substanceC0, accumulatedLeak)
        self.synapseParameters = sampleSynapseParameters(expectedMaximalTransmitterCount0,
                                                         varianceMaximalTransmitterCount0,
                                                         expectedMaximalRezeptorCount0,
                                                         varianceMaximalRezeptorCount0,
                                                         linearCliftLeak,
                                                         defaultReleasePercentage)
  #      self.sndOrderParameters = getHyperparameters(maxOfMaxVesicles=15000, maxOfMaxReceptors=12000,
   #                                                  minOfMaxVesicles=0, minOfMaxReceptors=0)

    def getLearningSignals(self):
        synapse = self.synapse
        learningSignals = synapse[]
        return learningSignals

    def step(self):
        self.synapse = synapseStep(self.synapseParameters, self.synapse)
        return self

    def updateSynapseParameters(self, newVetrikelMax, newRezeptorMax, newDefaultReleasePercentage):
        kTransBuild, vetrikelMax, kRezeptorRebuild, rezeptorMax, \
        kCollision, cliftLeakSpeed, linearCliftLeak, defaultReleasePercentage = self.synapseParameters
        self.synapseParameters = np.array([kTransBuild, newVetrikelMax, kRezeptorRebuild, newRezeptorMax,
                                            kCollision, cliftLeakSpeed, linearCliftLeak, newDefaultReleasePercentage])
        return self


    # Samples Parameters for an ineffective Connection
    def ineffectiveSynapse(self):
        self.synapseParameters = sampleSynapseParameters(1, 0, 1, 0, 100)
        return self

    # transmitters in the vesicles are released into the synaptic clift
    # the percentage defines which relative amount of transmitters is released with a spike in the preneuron
    def receiveSpike(self):
        t, g, r, i, releasePercentage, li = self.synapse
        newReleasePercentage = releasePercentageIncrease(releasePercentage)
        maximumNumberTransmittersInSynapticClift = MAX_CLIFT_TRANSMITTER_COUNT
        # it is not plausible, that there is a maximum amount of transmitters in the clift. There should be an exponential clift decay

        newVetrikelCount = np.ceil(t * (1 - releasePercentage))
        newCliftCount = np.floor(t * releasePercentage) + g
        if newCliftCount > maximumNumberTransmittersInSynapticClift:
            newCliftCount = maximumNumberTransmittersInSynapticClift
      #      print('Maximum number of transmitters reached in the synaptic clift! cliftTransmitters=', newCliftCount)
        self.synapse[0] = newVetrikelCount
        self.synapse[1] = newCliftCount
        self.synapse[4] = newReleasePercentage
        return self  # synapse  # np.array([newVetrikelCount, newCliftCount, r, i, newReleasePercentage, li])

    # postsynaptic learning indicators are released into the synaptic clift
    # the release percentage can also be used to measure recent activity
    # the building time of the substances could also be modeled, but at some point the complexity will grow to fast
    # returns a synapse with adjusted learningIndicators
    def receivePostsynapticSpike(self, staticSubstanceAPool=200, staticSubstanceCPool=200):
        t, g, freeReceptors, i, relativeRecentActivity, learningIndicators = self.synapse
        substanceA0, substanceB0, substanceC0, accumulatedLeak = learningIndicators
        kTransBuild, vetrikelMax, kRezeptorRebuild, rezeptorMax, \
        kCollision, cliftLeakSpeed, linearCliftLeak, defaultReleasePercentage = self.synapseParameters
        relativeAmountFreeReceptors = freeReceptors / rezeptorMax
        relativeSubstanceArelease = (relativeAmountFreeReceptors * relativeRecentActivity)
        relativeSubstanceCrelease = ((1 - relativeAmountFreeReceptors) * relativeRecentActivity)
        if relativeSubstanceCrelease < 0:
            relativeSubstanceCrelease = 0
        # print('relativeSubstanceArelease: ', relativeSubstanceArelease)
        #	print('relativeSubstanceCrelease: ', relativeSubstanceCrelease)
        # newLearningIndicators = np.array(
        #	[substanceA0 + np.round(staticSubstanceAPool * relativeSubstanceArelease), substanceB0,
        #	 substanceC0 + np.round(staticSubstanceCPool * relativeSubstanceCrelease)])
        learningIndicators[0] = substanceA0 + np.round(staticSubstanceAPool * relativeSubstanceArelease)
        learningIndicators[2] = substanceC0 + np.round(staticSubstanceCPool * relativeSubstanceCrelease)
        return self  # np.array([t, g, freeReceptors, i, relativeRecentActivity, newLearningIndicators])

    def __str__(self):
        self.vesiclesTransmitters = self.synapse[0]
        self.synapticCliftTransmitters = self.synapse[1]
        self.freeReceptors = self.synapse[2]
        self.acceptedTransmitters = self.synapse[3]
        self.releasePercentage = self.synapse[4]
        self.learningIndicators = self.synapse[5]

        self.substanceA = self.learningIndicators[0]
        self.substanceB = self.learningIndicators[1]
        self.substanceC = self.learningIndicators[2]

        self.maximumVesicles = self.synapseParameters[1]
        self.maximumReceptors = self.synapseParameters[3]
        self.cliftLeak = self.synapseParameters[6]
        self.cliftLeakSpeed = self.synapseParameters[5]
        self.defaultReleasePercentage = self.synapseParameters[7]
        self.transmitterBuildSpeed = self.synapseParameters[0]
        self.receptorFreeingSpeed = self.synapseParameters[2]
        self.receptorTransmitterCouplingSpeed = self.synapseParameters[4]
        return ('Transmitters bound in vesicles: ' + str(self.synapse[0]) + '\n' +
                'Transmitters in synaptic clift: ' + str(self.synapse[1]) + '\n' +
                'Free Rezeptors: ' + str(self.synapse[2]) + '\n' +
                'Accepted Transmitters: ' + str(self.synapse[3]) + '\n' +
                'Release Percentage: ' + str(self.synapse[4]) + '\n' +
                'Learning Signals: A=' + str(self.learningIndicators[0]) + ', B=' +
                str(self.learningIndicators[1]) + ', C=' + str(self.learningIndicators[2]) + '\n\n' +

                'Synapse Parameters:' +
                'Maximum Vesicles Amount: ' + str(self.synapseParameters[1]) + '\n' +
                'Maximum Receptor Amount: ' + str(self.synapseParameters[3]) + '\n' +
                'Synaptic Clift Leak Speed: ' + str(self.synapseParameters[5]) + '\n' +
                'Linear Synaptic Clift Leak: ' + str(self.synapseParameters[6]) + '\n' +
                'Default Release Percentage: ' + str(self.synapseParameters[7]) + '\n' +
                'Transmitter build speed: ' + str(self.synapseParameters[0]) + '\n' +
                'Receptor build speed: ' + str(self.synapseParameters[2]) + '\n' +
                'Transmitter-Receptor coupling rate: ' + str(self.synapseParameters[4]) + '\n\n\n')



# -------------------------Synapse Dynamics---------------------------------------
# Populationsdynamik
# This formula is worth criticizing, because slow growth resp. slow rebuilding, when few transmitter substances resp.
# receptors are available, makes no biological sense.

# This function models how the transmitters are rebuilt in the preneuronal vesicles / ventricles
# k needs to be larger the smaller the maximum value is
def transmitterBuildStep(k, maximum, synapse):
    transStatus, g, r, i, rp, li = synapse
    if transStatus < 0:
        transStatus = 1
    ntDenominator = (1 + (maximum / transStatus - 1) * np.exp(-k * maximum * 1))
    if ntDenominator == 0:
        ntDenominator = 0.00000000001
        print('Attempted division by zero in transmitterBuildStep')
    nt = maximum / ntDenominator  # maximum / (1 + (maximum/transStatus - 1) * np.exp(-k * maximum * 1))
    if nt < 0:
        print('Negative vesicles count: Something went wrong in the transmitterBuildStep function. vesicles=', nt)
        nt = 1
    if nt > maximum + 200:
        print('Too high vesicles count: Something went wrong in the transmitterBuildStep function. vesicles=', nt)
        nt = maximum

    if random.randint(0, 1):
        synapse[0] = np.ceil(nt)
        return synapse  # np.array([np.ceil(nt), g, r, i, rp, li])
    else:
        synapse[0] = np.floor(nt)
        return synapse  # np.array([np.floor(nt), g, r, i, rp, li])


# This function models how the percentage of released percent of transmitters in case of no arriving spike is decreasing over time
# Inspired by the change of temperature
def releasePercentageDecay(releasePercentage, defaultReleasePercentage=0.2, decaySpeed=0.008):
    return (np.exp(-decaySpeed) * (releasePercentage - defaultReleasePercentage) + defaultReleasePercentage)


# This function models how the percentage of released percent of transmitters in case of a spike is increasing
def releasePercentageIncrease(releasePercentage, maximumReleasePercentage=1, growthSpeed=0.9):
    return releasePercentageDecay(releasePercentage, maximumReleasePercentage, growthSpeed)


# models the release Percentage Decay of Transmitters when no information is transfered in a high frequency
def releasePercentageDecayStep(synapse, defaultReleasePercentage=0.2, releaseDecaySpeed=0.008):
    t, g, r, i, rp, li = synapse
    rp = releasePercentageDecay(rp, defaultReleasePercentage, releaseDecaySpeed)
    synapse[4] = rp
    return synapse  # np.array([t, g, r, i, rp, li])


# This function models how the receptors are rebuilt in the postneuron
# Inspired by the change of temperature
def rezeptorRebuildStep(k, maximum, synapse):
    t, g, rezeptorStatus, i, rp, li = synapse
    nr = np.exp(-k) * (rezeptorStatus - maximum) + maximum
    if random.randint(0, 1):
        synapse[2] = np.ceil(nr)
        return synapse  # np.array([t, g, np.ceil(nr), i, rp, li])
    else:
        synapse[2] = np.floor(nr)
        return synapse  # np.array([t, g, np.floor(nr), i, rp, li])


# Chemische Reaktionen zweiter Ordnung
# Models the speed of a reaction from type A + B -> C
# returns new values for A, B and C
# returns the next synapse state
def collisionDynamics(k, synapse):
    t, trans0, rezept0, i, rp, li = synapse
    # model the concentration of transmitters for one time step
    transCountNumerator = (trans0 * (trans0 - rezept0) * np.exp((trans0 - rezept0) * k * 1))
    transCountDenominator = (trans0 * np.exp((trans0 - rezept0) * k * 1) - rezept0)
    if transCountDenominator == 0:
        rezept0 = rezept0 + 1
        transCountNumerator = (trans0 * (trans0 - rezept0) * np.exp((trans0 - rezept0) * k * 1))
        transCountDenominator = (trans0 * np.exp((trans0 - rezept0) * k * 1) - rezept0)
    transCount = transCountNumerator / transCountDenominator
    # model the concentration of corresponding receptors for one time step
    rezeptCountNumerator = (rezept0 * (rezept0 - trans0) * np.exp((rezept0 - trans0) * k * 1))
    rezeptCountDenominator = (rezept0 * np.exp((rezept0 - trans0) * k * 1) - trans0)
    if rezeptCountDenominator == 0:
        trans0 = trans0 + 1
        rezeptCountNumerator = (rezept0 * (rezept0 - trans0) * np.exp((rezept0 - trans0) * k * 1))
        rezeptCountDenominator = (rezept0 * np.exp((rezept0 - trans0) * k * 1) - trans0)
    rezeptCount = rezeptCountNumerator / rezeptCountDenominator
    # calculate how many transmitters are allowed to enter the cell
    avgSuccess = np.floor((trans0 - transCount) + (rezept0 - rezeptCount) / 2)

    synapse[1] = np.ceil(transCount)
    synapse[2] = np.ceil(rezeptCount)
    synapse[3] = i + avgSuccess
    return synapse  # np.array([t, np.ceil(transCount), np.ceil(rezeptCount), i + avgSuccess, rp, li])


# at every time step the synaptic clift transmitters are leaking, so they become inefficient
# you need to change the synapse parameters and the synapseStep function if you want to use this linear leaking
def linearSynapseCliftLeak(synapticLeak, synapse):
    t, g, r, i, rp, li = synapse
    if g - synapticLeak <= 0:
        synapse[1] = 0
        return synapse  # np.array([t, 0, r, i, rp, li])
    else:
        synapse[1] = g - synapticLeak
        return synapse  # np.array([t, g - synapticLeak, r, i, rp, li])

# at every time step the synaptic clift transmitters are leaking, so they become inefficient
# the leak is faster the more transmitters are in the gap
# this introduces a soft maximum
def synapseCliftLeak(leakSpeed, synapse, minimum = 0):
    t, gapTransmitterCount, r, i, rp, learningIndicators = synapse
    ngt = np.exp(-leakSpeed) * (gapTransmitterCount - minimum) + minimum
    if random.randint(0, 1):
        synapse[1] = np.ceil(ngt)
        learningIndicators[3] = learningIndicators[3] + (gapTransmitterCount - np.ceil(ngt))
        synapse[5] = learningIndicators
        return synapse
    else:
        synapse[1] = np.floor(ngt)
        learningIndicators[3] = learningIndicators[3] + (gapTransmitterCount - np.floor(ngt))
        synapse[5] = learningIndicators
        return synapse

# calculates one step of synapse dynamics given a set of parameters
# it should be possible to calculate all those steps in parallel
# TODO could be run in parallel
def synapseStep(params, synapse):
    kTransBuild, vetrikelMax, kRezeptorRebuild, rezeptorMax, kCollision, cliftLeakSpeed, linearCliftLeak, defaultReleasePercentage = params
    synapse = transmitterBuildStep(kTransBuild, vetrikelMax, synapse)
    synapse = rezeptorRebuildStep(kRezeptorRebuild, rezeptorMax, synapse)
    synapse = collisionDynamicsLearningIndicatorsAtoB(synapse)
    synapse = releasePercentageDecayStep(synapse, defaultReleasePercentage, releaseDecaySpeed=0.008)
    synapse = substanceALeak(synapse)
    synapse = collisionDynamics(kCollision, synapse)
    synapse = synapseCliftLeak(cliftLeakSpeed, synapse, minimum=0)
    synapse = linearSynapseCliftLeak(linearCliftLeak, synapse)
    return synapse


# transmitters in the vesicles are released into the synaptic clift
# the percentage defines which relative amount of transmitters is released with a spike in the preneuron
def receiveSpike(synapse):
    t, g, r, i, releasePercentage, li = synapse
    newReleasePercentage = releasePercentageIncrease(releasePercentage)
    #	print('effective releasePercentage: ', releasePercentage)
    #	print('newReleasePercentage: ', newReleasePercentage)#, '\n')
    synapse[0] = np.ceil(t * (1 - releasePercentage))
    synapse[1] = np.floor(t * releasePercentage) + g
    synapse[4] = newReleasePercentage
    return synapse


# np.array([np.ceil(t * (1- releasePercentage)), np.floor(t * releasePercentage) + g, r, i, newReleasePercentage, li])


# -------------------------Define Synapse Parameters------------------------------

# Parameters for each synapse

'''
Influenced by the preneuron:
kTransBuild: defines the speed how fast the information can be send again, after it was sent. A low build up speed makes
the post neuron ignore high frequency firing of the preneuron.
vetrikelMax: defines the maximum intensity of the sent information. A higher maximal value increases the build up speed
of the information

Influenced by the postneuron:
kRezeptorRebuild: basically corresponds to the adaptation rate. Defines how fast the information is accepted. And how
quickly the postneuron accepts new information from this source. It modulates how fast a used receptor becomes
free after a spike.
rezeptorMax: defines the maximum intensity and speed with which information is accepted. A higher intensity leads to a
shorter active period in the post neuron though.

Modulatable by both preneuron and postneuron:
kCollision: basically corresponds to something like 1/size of the synaptic clift. This defines the probability of a
collision of transmitter and receptor. A higher speed uses more transmitter/time and receptors/time, but leads to a
faster and stronger effect in the post neuron.
cliftLeak: something like the opposite of an adaptation rate. Defines how much information is lost and ignored
per time step. This effect is more recognizable when the preneuron has a low maximum number of transmitters to release
or already fired often (does not release new transmitters) and when the rezeptor rebuild or freeing rate
is relatively small.
'''


# This function initializes randomly synapse parameters.
# You can influence the outcome of the sampling by adding expected values.
def sampleSynapseParameters(expectedMaximalTransmitterCount=2300, varianceMaximalTransmitterCount=100,
                            expectedMaximalRezeptorCount=1200, varianceMaximalRezeptorCount=100, linearCliftLeak = 4,
                            defaultReleasePercentage=None):
    vetrikelMax = np.ceil(
        np.absolute(np.random.logistic(expectedMaximalTransmitterCount, varianceMaximalTransmitterCount)))
    rezeptorMax = np.ceil(np.absolute(np.random.logistic(expectedMaximalRezeptorCount, varianceMaximalRezeptorCount)))
    if defaultReleasePercentage:
        defaultReleasePercentage = defaultReleasePercentage
    else:
        defaultReleasePercentage = np.random.uniform(0.05, 0.35)
    # TODO figure out a way to calibrate all those speeds

    kTransBuild = TRANSMITTER_BUILD_SPEED
    kRezeptorRebuild = RECEPTOR_BUILD_SPEED  # 0.007309 #0.006309 #0.0046109
    kCollision = COLLISION_SPEED

    cliftLeakSpeed = EXPONENTIAL_CLIFT_LEAK_SPEED  #0.00001#0.095
    linearCliftLeak = linearCliftLeak

    #kTransBuild = 0.00001386294361119890618834464242916353136151
    #kRezeptorRebuild = 0.0109
    #kCollision = 0.00001386294361119890618834464242916353136151

    synapseParameters = np.array(
        [kTransBuild, vetrikelMax, kRezeptorRebuild, rezeptorMax, kCollision, cliftLeakSpeed, linearCliftLeak, defaultReleasePercentage])
    return synapseParameters


# If Spiking postneuron excitory rezeptors are free, this means that this synapse did not participate in the spiking.
# Maybe a substance A should be released by the postneuron, if it spikes and the rezeptors are not used.
# So the release percantage of substance A can be given by the relative amount of free rezeptors
# w.r.t. amount of receptors in the synapse.
# If the synapse is used while the substance A is present, the synapse effect should be decreased.
# Basically substance A and the excitory transmitters react to substance B, which indicates that the connection
# intensity should be decreased.
# If the excitory rezeptors are blocked on the other hand, when a postneuron spikes, then this synapse
# participated in the spiking
# A substance C should be released, which indicates that the synapse should be enforced.
# In conclusion:
# if substance B is present the connections intensity is decreased
# if substance C is present the connections intensity is increased

# Substance A has to leak, because we only want to modulate the connection, when the preneural spike arrives
# shortly after the postneural spike
# Substances B and C don't leak, but are reseted, when the learning step is performed.

# The same substances have the opposite effect on inhibitory connections, so
# if substance B is present the connections intensity is increased
# if substance C is present the connections intensity is decreased


# This function has to be called before the collision dynamics
# Reaktionen zweiter Ordnung
# Models the speed of a reaction from type substance A + transmitter -> substance B
# returns new values for substance A, transmitter and substance B
# returns the next synapse state
# k corresponds to the time difference sensitivity
def collisionDynamicsLearningIndicatorsAtoB(synapse,
                                            timeDifferenceSensitivity=TIME_DIFFERENCE_SENSITIVITY_LEARNING_SIGNAL):  # 0.0000138629):
    learningAffectsInformationTransfer = True
    k = timeDifferenceSensitivity
    t, trans0, r, i, rp, learningIndicators = synapse
    substanceA0, substanceB0, substanceC0, accumulatedLeak = learningIndicators

    # model the concentration of transmitters for one time step
    transCountNumerator = (trans0 * (trans0 - substanceA0) * np.exp((trans0 - substanceA0) * k * 1))
    transCountDenominator = (trans0 * np.exp((trans0 - substanceA0) * k * 1) - substanceA0)
    if transCountDenominator == 0:
        trans0 = trans0 + 1
        transCountNumerator = (trans0 * (trans0 - substanceA0) * np.exp((trans0 - substanceA0) * k * 1))
        transCountDenominator = (trans0 * np.exp((trans0 - substanceA0) * k * 1) - substanceA0)
    transCount = np.round(transCountNumerator / transCountDenominator)
    # model the concentration of substance A for one time step
    substanceACountNumerator = (substanceA0 * (substanceA0 - trans0) * np.exp((substanceA0 - trans0) * k * 1))
    substanceACountDenominator = (substanceA0 * np.exp((substanceA0 - trans0) * k * 1) - trans0)
    if substanceACountDenominator == 0:
        substanceA0 = substanceA0 + 1
        substanceACountNumerator = (substanceA0 * (substanceA0 - trans0) * np.exp((substanceA0 - trans0) * k * 1))
        substanceACountDenominator = (substanceA0 * np.exp((substanceA0 - trans0) * k * 1) - trans0)
    substanceACount = np.round(substanceACountNumerator / substanceACountDenominator)
    # calculate how many transmitters are allowed to enter the cell
    substanceBsynthesised = np.round((trans0 - transCount) + (substanceA0 - substanceACount) / 2)

    #	newLearningIndicators = [substanceACount, np.round(substanceB0 + substanceBsynthesised), substanceC0]
    learningIndicators[0] = substanceACount
    learningIndicators[1] = np.round(substanceB0 + substanceBsynthesised)
    if learningAffectsInformationTransfer:
        synapse[1] = np.ceil(transCount)
        synapse[5] = learningIndicators
        return synapse
    else:  # this case does not make sense
        synapse[1] = trans0
        synapse[5] = learningIndicators  # newLearningIndicators
    # nextSynapse = np.array([t, trans0, r, i, rp, newLearningIndicators])
    return synapse  # nextSynapse


# models the exponential leak of substance A until it reaches zero
def substanceALeak(synapse, leakSpeed=0.02):
    t, trans0, r, i, rp, learningIndicators = synapse
    substanceA0, substanceB0, substanceC0, accumulatedLeak = learningIndicators
    substanceA = np.round(np.exp(-leakSpeed) * (substanceA0))

    learningIndicators[0] = substanceA
    synapse[5] = learningIndicators
    return synapse



# -------------------------Define Learning of Synapse Parameters--------------


