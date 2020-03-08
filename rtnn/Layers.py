import numpy as np

from . import neuron as n
from . import synapse as s

from . import errors as err

# Initializes a vector of neurons in one layer
def initLayer(numberOfNeurons):
    neurons = []
    for i in range(0, numberOfNeurons):
        neurons.append(n.Neuron())
    layer = np.array(neurons)
    return layer


class Layer:
    def __init__(self, numberOfNeurons):
        self.layer = initLayer(numberOfNeurons)

    def __len__(self):
        return len(self.layer)

    def __getitem__(self, item):
        return self.layer[item]

    def __setitem__(self, key, value):
        self.layer[key] = value
        return self

    def __delitem__(self, key):
        self.layer.__delitem__(key)
        return self

    def __iter__(self):
        return self.layer.__iter__()



class InputLayer:
    def __init__(self, numberOfInputNeurons):
        self.preneurons = Layer(numberOfInputNeurons)
        self.postneurons = self.preneurons

    def __len__(self):
        return len(self.preneurons)

    def __str__(self):
        layerDecription = ''
        for neuronIx, neuron in enumerate(self.postneurons):
            layerDecription = layerDecription + 'Neuron ' + str(neuronIx) + ':\n' + str(neuron)
        return layerDecription

    # makes one step for a layer of neurons with parameters, that receive a new list of input potentials
    # returns the new state of the layer of neurons
    # exitoryInhibitoryLayerDynamics(neuronParameters, neurons, exitoryPotentials, inhibitoryPotentials)
    def step(self, excitatoryInputs, inhibitoryInputs):

        if len(excitatoryInputs) != len(self.preneurons):
            msg = 'Preneurons have length ' + str(len(self.preneurons)) + \
                  ', but preneural exitory input has shape ' + str(excitatoryInputs.shape)
            raise err.InputPotentialError(msg)
        if len(inhibitoryInputs) != len(self.preneurons):
            msg = 'Preneurons have length ' + str(len(self.preneurons)) + \
                  ', but preneural inhibitory input has shape ' + str(inhibitoryInputs.shape)
            raise err.InputPotentialError(msg)

        # TODO this loop could be parallelised
        for index in range(0, len(self.preneurons)):
            newExcitatoryPotential = excitatoryInputs[index]
            newInhibitoryPotential = inhibitoryInputs[index]
            neuron = self.preneurons.layer[index]
            neuron.step(newExcitatoryPotential, newInhibitoryPotential)
        return self


# Initializes a matrix of synapses fully connecting two layers of neurons
def initFullyConnectedLayer(preNeurons, postNeurons,
                            expectedMaximalTransmitterCount0=3000,varianceMaximalTransmitterCount0=10,
                            expectedMaximalRezeptorCount0=1200, varianceMaximalRezeptorCount0=10,
                            defaultReleasePercentage=None):
        synapseMatrix = np.full((len(preNeurons), len(postNeurons)), s.Synapse())
        # each synapse needs to be generated on its own to get an unambiguous address
        for i in range(len(preNeurons)):
            for j in range(len(postNeurons)):
                synapseMatrix[i,j] = s.Synapse(expectedMaximalTransmitterCount0,
                                               varianceMaximalTransmitterCount0,
                                               expectedMaximalRezeptorCount0,
                                               varianceMaximalRezeptorCount0)
        return synapseMatrix


class SynapseMatrix:
    def __init__(self, preneurons, postneurons,
                 expectedMaximalTransmitterCount0=3000, varianceMaximalTransmitterCount0=100,
                 expectedMaximalRezeptorCount0=1200, varianceMaximalRezeptorCount0=100):
        self.synapseMatrix = initFullyConnectedLayer(preneurons, postneurons,
                                                     expectedMaximalTransmitterCount0, varianceMaximalTransmitterCount0,
                                                     expectedMaximalRezeptorCount0, varianceMaximalRezeptorCount0)

    # Tells you how many pre-neurons and post-neurons are connected by a synapse matrix
    def getNumberOfConnectedNeurons(self):
        noPreNeurons = len(self.synapseMatrix)
        noPostNeurons = len(self.synapseMatrix[0])
        return noPreNeurons, noPostNeurons

    def step(self):
        preNo, postNo = self.getNumberOfConnectedNeurons()
        # TODO loop could be run in parallel
        for preIndex in range(0, preNo):
            for postIndex in range(0, postNo):
                synapse = self.synapseMatrix[preIndex][postIndex]
                self.synapseMatrix[preIndex][postIndex] = synapse.step()
        return self

    # returns the synapse matrix after a specific neuron spiked
    # function does not check if a spike occured in the preneuron
    # this function is called after a preneuron specified by an index spiked
    def preNeuronSpike(self, preNeuronIndex):
        preNo, postNo = self.getNumberOfConnectedNeurons()
        # TODO loop could be run in parallel
        for postIndex in range(0, postNo):
            synapseToReceive = self.synapseMatrix[preNeuronIndex, postIndex]
            self.synapseMatrix[preNeuronIndex, postIndex] = synapseToReceive.receiveSpike()
        return self

    # receive current state of preneurons and decides if to returns a new synapse matrix if there was some activity
    def triggerIfAdequat(self, preneurons):
        # TODO loop could be run in parallel
        for index in range(0, len(preneurons)):
            inputExcitoryPotential, inputInhibitoryPotential, remainingRefractoryTime, \
            spiking, neuronLearningSignal = preneurons[index].neuron
            if spiking:
                self.preNeuronSpike(index) # TODO Check if this is enough to keep the next state
        return self

    # returns the synapse matrix connecting to the previous layer after a specific neuron spiked
    # function does not check if a spike occured in the postneuron
    # this function is called after a postneuron specified by an index spiked
    def postNeuronSpike(self, postNeuronIndex):
        preNo, postNo = self.getNumberOfConnectedNeurons()
        # TODO loop could be run in parallel
        for preIndex in range(0, preNo):
            preSynapseToReceive = self.synapseMatrix[preIndex, postNeuronIndex]
            self.synapseMatrix[preIndex, postNeuronIndex] = preSynapseToReceive.receivePostsynapticSpike()
        return self

    # receive current state of preneurons and decides if to returns a new synapse matrix if there was some activity
    # Releases learning indicators
    def triggerPostSynapseMatrixIfAdequat(self, postneurons):
        # TODO loop could be run in parallel
        for postIndex in range(0, len(postneurons)):
            inputExcitoryPotential, inputInhibitoryPotential, remainingRefractoryTime, spiking, neuronLearningSignal = \
            postneurons[postIndex].neuron
            if spiking:
                self.postNeuronSpike(postIndex)
        return self

    # deletes refelxive connections, so a neuron cannot inhibit itself
    def asLateralInhibitionMatrix(self):
        preNo, postNo = self.getNumberOfConnectedNeurons()
        for i in range(postNo):
            reflexiveSynapse = self.synapseMatrix[i][i]
            reflexiveSynapse.ineffectiveSynapse()

        return self

    # returns the new input potential for postNeuronIndex and the new synapse matrix
    def integrateInputPotentialOf(self, postNeuronIx):
        vGetAcceptedTransmitters = np.vectorize(s.getAcceptedTransmitters)
        vResetAcceptedTransmitters = np.vectorize(s.resetAcceptedTransmitters)
        inputPotential = np.sum(vGetAcceptedTransmitters(self.synapseMatrix[:, postNeuronIx]))
        vResetAcceptedTransmitters(self.synapseMatrix[:, postNeuronIx])
        return inputPotential, self






class LateralInhibitoryLayer:
    def __init__(self, numberOfPostneurons, excitatoryTransmitterExpectation=5000, excitatoryTransmitterVariance=10,
              excitatoryReceptorExpectation=1500, excitatoryReceptorVariance=10,
              inhibitoryTransmitterExpectation=3500, inhibitoryTransmitterVariance=10,
              inhibitoryReceptorExpectation=1200, inhibitoryReceptorVariance=10,
              lateralInhibitoryTransmitterExpectation=3500, lateralInhibitoryTransmitterVariance=10,
              lateralInhibitoryReceptorExpectation=1200, lateralInhibitoryReceptorVariance=10):
        self.postneurons = Layer(numberOfPostneurons)
        self.lateralInhibitoryMatrix = SynapseMatrix(self.postneurons, self.postneurons,
                                                     expectedMaximalTransmitterCount0=lateralInhibitoryTransmitterExpectation,
                                                     varianceMaximalTransmitterCount0=lateralInhibitoryTransmitterVariance,
                                                     expectedMaximalRezeptorCount0=lateralInhibitoryReceptorExpectation,
                                                     varianceMaximalRezeptorCount0=lateralInhibitoryReceptorVariance)
        self.lateralInhibitoryMatrix.asLateralInhibitionMatrix()

        self.buildParameters = np.array([excitatoryTransmitterExpectation, excitatoryTransmitterVariance,
              excitatoryReceptorExpectation, excitatoryReceptorVariance,
              inhibitoryTransmitterExpectation, inhibitoryTransmitterVariance,
              inhibitoryReceptorExpectation, inhibitoryReceptorVariance])

    # build is basically the second part of the __init__ method, but we can call it later,
    # when we know what input the layer receives
    def build(self, inputLayer):
        excitatoryTransmitterExpectation, excitatoryTransmitterVariance,\
        excitatoryReceptorExpectation, excitatoryReceptorVariance,\
        inhibitoryTransmitterExpectation, inhibitoryTransmitterVariance, \
        inhibitoryReceptorExpectation, inhibitoryReceptorVariance = self.buildParameters
        self.excitatoryMatrix = SynapseMatrix(inputLayer, self.postneurons,
                                              expectedMaximalTransmitterCount0=excitatoryTransmitterExpectation,
                                              varianceMaximalTransmitterCount0=excitatoryTransmitterVariance,
                                              expectedMaximalRezeptorCount0=excitatoryReceptorExpectation,
                                              varianceMaximalRezeptorCount0=excitatoryReceptorVariance)
        self.inhibitoryMatrix = SynapseMatrix(inputLayer, self.postneurons,
                                              expectedMaximalTransmitterCount0=inhibitoryTransmitterExpectation,
                                              varianceMaximalTransmitterCount0=inhibitoryTransmitterVariance,
                                              expectedMaximalRezeptorCount0=inhibitoryReceptorExpectation,
                                              varianceMaximalRezeptorCount0=inhibitoryReceptorVariance)

    def __len__(self):
        return len(self.postneurons)

    def __str__(self):
        layerDecription = ''
        for neuronIx, neuron in enumerate(self.postneurons):
            layerDecription = layerDecription + 'Neuron ' + str(neuronIx) + ':\n' + str(neuron)
        return layerDecription

    # perform one time step for an input layer or an inner layer of the neural network
    # inhibits other neurons in the same layer after a spike arrived
    # needs the neurons of the preceding layer to run: preneurons
    # Biases are injectable Potentials, e.g. to produce a desired output
    # ignorePreneurons regulates if the bias is combined with the normal activity (False) or
    # becomes the only input for the postneurons (True)
    def step(self, preneurons, learning = True, exitatoryBiases = None, inhibitoryBiases = None, ignorePreneurons = False):

        if (exitatoryBiases is None) and (inhibitoryBiases is None):
            ignorePreneurons = False

        if not (exitatoryBiases is None):
            if len(exitatoryBiases) != len(self.postneurons):
                msg = 'Postneurons have length ' + str(len(self.postneurons)) + \
                      ', but postneural exitory bias have length ' + str(len(exitatoryBiases)) #str(exitatoryBiases.shape)
                raise err.InputPotentialError(msg)

        if not (inhibitoryBiases is None):
            if len(inhibitoryBiases) != len(self.postneurons):
                msg = 'Postneurons have length ' + str(len(self.postneurons)) + \
                      ', but postneural inhibitory bias have length ' + str(len(inhibitoryBiases))
                raise err.InputPotentialError(msg)

        # TODO these functions could be executed in parallel
        self.excitatoryMatrix = self.excitatoryMatrix.triggerIfAdequat(preneurons)
        self.inhibitoryMatrix = self.inhibitoryMatrix.triggerIfAdequat(preneurons)
        self.lateralInhibitoryMatrix = self.lateralInhibitoryMatrix.triggerIfAdequat(self.postneurons)

        if learning: # collecting the learning indicators
            self.excitatoryMatrix = self.excitatoryMatrix.triggerPostSynapseMatrixIfAdequat(self.postneurons)
            self.inhibitoryMatrix = self.inhibitoryMatrix.triggerPostSynapseMatrixIfAdequat(self.postneurons)
            self.lateralInhibitoryMatrix = self.lateralInhibitoryMatrix.triggerPostSynapseMatrixIfAdequat(
                self.postneurons)

        # calculating the synapse dynamics
        self.excitatoryMatrix = self.excitatoryMatrix.step()
        self.inhibitoryMatrix = self.inhibitoryMatrix.step()
        self.lateralInhibitoryMatrix = self.lateralInhibitoryMatrix.step()

        # TODO loop could be run in parallel
        for postNeuronIndex in range(0, len(self.postneurons)):
            postExitoryPotential, self.excitatoryMatrix = \
                self.excitatoryMatrix.integrateInputPotentialOf(postNeuronIndex)
            postInhibitoryPotential, self.inhibitoryMatrix = \
                self.inhibitoryMatrix.integrateInputPotentialOf(postNeuronIndex)
            additionalPostInhibitoryPotential, self.lateralInhibitoryMatrix = \
                self.lateralInhibitoryMatrix.integrateInputPotentialOf(postNeuronIndex)
            postInhibitoryPotential = postInhibitoryPotential + additionalPostInhibitoryPotential

            if not (exitatoryBiases is None):
                if ignorePreneurons:
                    postExitoryPotential = exitatoryBiases[postNeuronIndex]
                else:
                    postExitoryPotential = postExitoryPotential + exitatoryBiases[postNeuronIndex]

            if not (inhibitoryBiases is None):
                if ignorePreneurons:
                    postInhibitoryPotential = inhibitoryBiases[postNeuronIndex]
                else:
                    postInhibitoryPotential = postInhibitoryPotential + inhibitoryBiases[postNeuronIndex]

            pivotNeuron = self.postneurons[postNeuronIndex]
            self.postneurons[postNeuronIndex] = pivotNeuron.step(postExitoryPotential, postInhibitoryPotential)
        # self.postneurons is the input for the next layer
        return self


'''
# Initializes and returns three matrices of synapses.
# The first one is supposed to be the exitory synapse matrix
# The second one is supposed to be the inhibitory synapse matrix
# The third one is supposed to be lateral inhibitory synapse matrix 
def initLateralInhibitoryLayer(preNeurons, postNeurons):
    exitorySynapseMatrix = np.full((len(preNeurons), len(postNeurons), 6), S.Synapse()) # initSynapse(1, 0, 1, 0, 0.6, 0, 0, 0))
    inhibitorySynapseMatrix = np.full((len(preNeurons), len(postNeurons), 6), S.Synapse()) #initSynapse(1, 0, 1, 0, 0.6, 0, 0, 0))
    lateralInhibitorySynapseMatrix = np.full((len(postNeurons), len(postNeurons), 6), S.Synapse()) # initSynapse(1, 0, 1, 0, 0.6, 0, 0, 0))
    return exitorySynapseMatrix, inhibitorySynapseMatrix, lateralInhibitorySynapseMatrix
'''



# -------------------------Synapse Matrix Dynamics---------------------------------------
'''
# uses matrix of parameters and corresponding matrix of synapses to calculate the next step of the synapse matrix
def synapseMatrixStep(parameterMatrix, synapseMatrix):
    if ((len(synapseMatrix.flatten())) / 6) != ((len(parameterMatrix.flatten())) / 7):
        msg = 'Synapse matrix has shape ' + str(synapseMatrix.shape) + ', but parameters has shape ' + str(parameterMatrix.shape)
        raise E.SynapseParameterError(msg)
    preNo, postNo = getNumberOfConnectedNeurons(synapseMatrix)
    # loop could be run in parallel
    for preIndex in range(0, preNo):
        for postIndex in range(0, postNo):
            synapseMatrix[preIndex][postIndex] = synapseStep(parameterMatrix[preIndex][postIndex], synapseMatrix[preIndex][postIndex])
        #    (synapseMatrix[preIndex][postIndex]).step
    return synapseMatrix
'''

'''
# returns the new input potential for postNeuronIndex and the new synapse matrix
def integrateInputPotentialOf(synapseMatrix, neuronIx):
    inputPotential = np.sum(synapseMatrix[:,neuronIx,3])
    synapseMatrix[:,neuronIx,3] = 0
    return inputPotential, synapseMatrix
'''

'''
# makes one step for a layer of neurons with parameters, that receive a new list of input potentials
# returns the new state of the layer of neurons
def exitoryInhibitoryLayerDynamics(neuronParameters, neurons, exitoryPotentials, inhibitoryPotentials):
    for index in range(0, len(neurons)):
        threshold, spikeValue, leak, refractoryTime, maximalInhibition, inhibitionLeak = neuronParameters[index]
        newExitoryPotential = exitoryPotentials[index]
        newInhibitoryPotential = inhibitoryPotentials[index]
        neurons[index] = N.exitoryInhibitoryNeuronDynamics(threshold, spikeValue, leak, refractoryTime, maximalInhibition,
                                                         inhibitionLeak, neurons, index, newExitoryPotential,
                                                         newInhibitoryPotential)
    return neurons
'''



'''
# returns the synapse matrix after a specific neuron spiked
# function does not check if a spike occured in the preneuron
# this function is called after a preneuron specified by an index spiked
def preNeuronSpike(synapseMatrix, preNeuronIndex):
    preNo, postNo = getNumberOfConnectedNeurons(synapseMatrix)
    for postIndex in range(0, postNo):
        synapseMatrix[preNeuronIndex, postIndex] = receiveSpike(synapseMatrix[preNeuronIndex, postIndex])
    return synapseMatrix


# receive current state of preneurons and decides if to returns a new synapse matrix if there was some activity
def synapseMatrixTriggerIfAdequat(preneurons, synapseMatrix):
    for index in range(0, len(preneurons)):
        inputExcitoryPotential, inputInhibitoryPotential, remainingRefractoryTime, \
        spiking, neuronLearningSignal = preneurons[index]
        if spiking:
            synapseMatrix = preNeuronSpike(synapseMatrix, index)
    return synapseMatrix
'''

'''
# perform one time step for an input layer or an inner layer of the neural network
def exitoryInhibitoryLayerTimeStep(preneuralExcitoryInputs, preneuralInhibitoryInputs, preneurons,
                                   preNeuronParameterList, postneurons, exitorySynapseMatrix,
                                   exitorySynapseParameterMatrix, inhibitorySynapseMatrix,
                                   inhibitorySynapseParameterMatrix):
    if len(preneuralExcitoryInputs) != len(preneurons):
        msg = 'Preneurons have shape ' + str(preneurons.shape) + ', but preneural exitory input has shape ' + str(
            preneuralExcitoryInputs.shape)
        raise InputPotentialError(msg)
    if len(preneuralInhibitoryInputs) != len(preneurons):
        msg = 'Preneurons have shape ' + str(preneurons.shape) + ', but preneural inhibitory input has shape ' + str(
            preneuralInhibitoryInputs.shape)
        raise InputPotentialError(msg)
    # calculate the internal neuron dynamics for the preneurons
    preneurons = exitoryInhibitoryLayerDynamics(preNeuronParameterList, preneurons, preneuralExcitoryInputs,
                                                preneuralInhibitoryInputs)
    # calculating the synapse dynamics
    exitorySynapseMatrix = synapseMatrixTriggerIfAdequat(preneurons, exitorySynapseMatrix)
    exitorySynapseMatrix = synapseMatrixStep(exitorySynapseParameterMatrix, exitorySynapseMatrix)
    inhibitorySynapseMatrix = synapseMatrixTriggerIfAdequat(preneurons, inhibitorySynapseMatrix)
    inhibitorySynapseMatrix = synapseMatrixStep(inhibitorySynapseParameterMatrix, inhibitorySynapseMatrix)
    # collect the results of the matrix steps and prepare the post neurons, such that post neural dynamics can be calculated
    postExitoryPotentialList = []
    postInhibitoryPotentialList = []
    for postNeuronIndex in range(0, len(postneurons)):
        postExitoryNeuralInput, exitorySynapseMatrix = integrateInputPotentialOf(exitorySynapseMatrix, postNeuronIndex)
        postInhibitoryNeuralInput, inhibitorySynapseMatrix = integrateInputPotentialOf(inhibitorySynapseMatrix,
                                                                                       postNeuronIndex)
        postExitoryPotentialList.append(postExitoryNeuralInput)
        postInhibitoryPotentialList.append(postInhibitoryNeuralInput)
    postExitoryPotentials = np.array(postExitoryPotentialList)
    postInhibitoryPotentials = np.array(postInhibitoryPotentialList)
    return preneurons, exitorySynapseMatrix, inhibitorySynapseMatrix, postExitoryPotentials, postInhibitoryPotentials
'''

'''
# perform one time step for an input layer or an inner layer of the neural network
def layerTimeStepWithLearning(preneuralExcitoryInputs, preneuralInhibitoryInputs, preneurons, preNeuronParameterList,
                              postneurons, exitorySynapseMatrix, exitorySynapseParameterMatrix, inhibitorySynapseMatrix,
                              inhibitorySynapseParameterMatrix):  # , exitoryPreSynapseMatrix, exitoryPreSynapseParameterMatrix, inhibitoryPreSynapseMatrix, inhibitoryPreSynapseParameterMatrix):
    if len(preneuralExcitoryInputs) != len(preneurons):
        msg = 'Preneurons have shape ' + str(preneurons.shape) + ', but preneural exitory input has shape ' + str(
            preneuralExcitoryInputs.shape)
        raise InputPotentialError(msg)
    if len(preneuralInhibitoryInputs) != len(preneurons):
        msg = 'Preneurons have shape ' + str(preneurons.shape) + ', but preneural inhibitory input has shape ' + str(
            preneuralInhibitoryInputs.shape)
        raise InputPotentialError(msg)
    # calculate the internal neuron dynamics for the preneurons
    preneurons = exitoryInhibitoryLayerDynamics(preNeuronParameterList, preneurons, preneuralExcitoryInputs,
                                                preneuralInhibitoryInputs)
    # collecting the learning indicators
    exitorySynapseMatrix = triggerPostSynapseMatrixIfAdequat(postneurons, exitorySynapseMatrix,
                                                             exitorySynapseParameterMatrix)
    inhibitorySynapseMatrix = triggerPostSynapseMatrixIfAdequat(postneurons, inhibitorySynapseMatrix,
                                                                inhibitorySynapseParameterMatrix)
    # collecting foreward information
    exitorySynapseMatrix = synapseMatrixTriggerIfAdequat(preneurons, exitorySynapseMatrix)
    inhibitorySynapseMatrix = synapseMatrixTriggerIfAdequat(preneurons, inhibitorySynapseMatrix)
    # calculating the synapse dynamics
    exitorySynapseMatrix = synapseMatrixStep(exitorySynapseParameterMatrix, exitorySynapseMatrix)
    inhibitorySynapseMatrix = synapseMatrixStep(inhibitorySynapseParameterMatrix, inhibitorySynapseMatrix)
    # collect the results of the matrix steps and prepare the post neurons, such that post neural dynamics can be calculated
    postExitoryPotentialList = []
    postInhibitoryPotentialList = []
    for postNeuronIndex in range(0, len(postneurons)):
        postExitoryNeuralInput, exitorySynapseMatrix = integrateInputPotentialOf(exitorySynapseMatrix, postNeuronIndex)
        postInhibitoryNeuralInput, inhibitorySynapseMatrix = integrateInputPotentialOf(inhibitorySynapseMatrix,
                                                                                       postNeuronIndex)
        postExitoryPotentialList.append(postExitoryNeuralInput)
        postInhibitoryPotentialList.append(postInhibitoryNeuralInput)
    postExitoryPotentials = np.array(postExitoryPotentialList)
    postInhibitoryPotentials = np.array(postInhibitoryPotentialList)
    return preneurons, exitorySynapseMatrix, inhibitorySynapseMatrix, postExitoryPotentials, postInhibitoryPotentials
'''

'''
# inhibits other neurons in the same layer after a spike arrived
# returns the adjusted postneural inhibition and the next inhibition synapse state
# can be used to adjust the inhibitory input for the next layer
def lateralInhibitoryLayerTimeStep(postneuralInhibitoryInputs, postneurons, lateralInhibitorySynapseMatrix,
                                   lateralInhibitorySynapseMatrixParameters):
    # collecting the learning indicators
    lateralInhibitorySynapseMatrix = triggerPostSynapseMatrixIfAdequat(postneurons, lateralInhibitorySynapseMatrix,
                                                                       lateralInhibitorySynapseMatrixParameters)
    lateralInhibitorySynapseMatrix = synapseMatrixTriggerIfAdequat(postneurons, lateralInhibitorySynapseMatrix)
    lateralInhibitorySynapseMatrix = synapseMatrixStep(lateralInhibitorySynapseMatrixParameters,
                                                       lateralInhibitorySynapseMatrix)
    for postNeuronIndex in range(0, len(postneurons)):
        additionalPostInhibitoryNeuralInput, lateralInhibitorySynapseMatrix = integrateInputPotentialOf(
            lateralInhibitorySynapseMatrix, postNeuronIndex)
        postneuralInhibitoryInputs[postNeuronIndex] = postneuralInhibitoryInputs[
                                                          postNeuronIndex] + additionalPostInhibitoryNeuralInput
    return postneuralInhibitoryInputs, lateralInhibitorySynapseMatrix
'''

'''
# perform one time step of the last layer. Basically decide, if the neurons are spiking or not.
def exitoryInhibitoryOutputLayerTimeStep(postneuralExcitoryInputs, postneuralInhibitoryInputs, postneurons,
                                         postNeuronParameterList):
    postneurons = exitoryInhibitoryLayerDynamics(postNeuronParameterList, postneurons, postneuralExcitoryInputs,
                                                 postneuralInhibitoryInputs)
    return postneurons
'''

# -------------------------Define Synapse Matrix Parameters------------------------------

'''
# Initializes a matrix of synapse parameters for a fully connected layer between preneurons and postneurons
def sampleSynapseParameterMatrix(preNeurons, postNeurons, expectedMaximalTransmitterCount = 2300, varianceMaximalTransmitterCount = 100, expectedMaximalRezeptorCount = 1200, varianceMaximalRezeptorCount = 100, expectedLeak = 2, varianceLeak = 2, defaultReleasePercentage = None):
    synapseParamMatrix = np.full((len(preNeurons), len(postNeurons), 7), sampleSynapseParameters())
    for i in range(0,len(preNeurons)):
        for j in range(0,len(postNeurons)):
            synapseParamMatrix[i][j] = sampleSynapseParameters(expectedMaximalTransmitterCount, varianceMaximalTransmitterCount, expectedMaximalRezeptorCount, varianceMaximalRezeptorCount, expectedLeak, varianceLeak, defaultReleasePercentage)
    return synapseParamMatrix

# Returns a matrix of synapse parameters to influence (inhibit) neighbouring neurons
def sampleLateralInhibitonMatrix(layerNeurons, expectedMaximalTransmitterCount = 2300, varianceMaximalTransmitterCount = 100, expectedMaximalRezeptorCount = 1200, varianceMaximalRezeptorCount = 100, expectedLeak = 2, varianceLeak = 2, defaultReleasePercentage = None):
    synapseParamMatrix = np.full((len(layerNeurons), len(layerNeurons), 7), sampleSynapseParameters())
    for i in range(0,len(layerNeurons)):
        for j in range(0,len(layerNeurons)):
            synapseParamMatrix[i][j] = sampleSynapseParameters(expectedMaximalTransmitterCount, varianceMaximalTransmitterCount, expectedMaximalRezeptorCount, varianceMaximalRezeptorCount, expectedLeak, varianceLeak, defaultReleasePercentage)
    for i in range(0, len(layerNeurons)):
        synapseParamMatrix[i][i] = noConnectionSynapseParameters()
    return synapseParamMatrix
'''

