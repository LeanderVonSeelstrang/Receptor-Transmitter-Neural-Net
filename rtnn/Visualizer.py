from . import Layers as L
# from . import network # as Net

import matplotlib.pyplot as plt
import numpy as np

def visualizeNeuronPotentials(network, neuronHistory):

    neuronList = []
    for ix, layer in enumerate(network.layers):
        maxNeurons = (ix, len(layer.postneurons))
        neuronList.append(maxNeurons)
    numNeurons = np.array(neuronList)
    numberLayers = len(neuronHistory[0])
    maxNeuronsPerLayer = np.max(numNeurons[:,1])

    t = np.linspace(0, len(neuronHistory), len(neuronHistory))
    fig, axs = plt.subplots(numberLayers, maxNeuronsPerLayer,
                            sharex='all', sharey='all', figsize=(maxNeuronsPerLayer * 2, numberLayers * 2))

    for layerIx, layer in enumerate(network.layers):
        for neuronId in range(numNeurons[layerIx,1]):
            layerHistory = np.array(neuronHistory[:, layerIx])
            excitatoryData = []
            inhibitoryData = []
            for time in range(len(layerHistory)):
                excitatoryData.append(layerHistory[time][neuronId].neuron[0])
                inhibitoryData.append(layerHistory[time][neuronId].neuron[1])

            excitatoryData = np.array(excitatoryData)
            inhibitoryData = np.array(inhibitoryData)

            title = 'Hidden neuron ' + str(neuronId)
            if layerIx == 0:
                title = 'Input neuron ' + str(neuronId)
            if layerIx == network.numLayers - 1: #len(layer) - 1:
                title = 'Output neuron ' + str(neuronId)

            if numNeurons[layerIx, 1] > 1:
                axs[layerIx, neuronId].set_title(title)
                line1, = axs[layerIx, neuronId].plot(t, excitatoryData, dashes=[6, 2], label='Neuron Potential')
                line2, = axs[layerIx, neuronId].plot(t, inhibitoryData, dashes=[4, 2, 8, 3], label='Neuron Inhibition')
                axs[layerIx, neuronId].set_ylabel('potential')
              #  axs[layerIx, neuronId].legend()

                if neuronId == numNeurons[layerIx,1]:
                    axs[layerIx, neuronId].set_xlabel('time')

                if layerIx == 0 and neuronId == 0:
                    axs[layerIx, neuronId].legend()

            else: # there is only one neuron in this layer
                axs[layerIx].set_title(title)
                line1, = axs[layerIx].plot(t, excitatoryData, dashes=[6, 2], label='Neuron Potential')
                line2, = axs[layerIx].plot(t, inhibitoryData, dashes=[4, 2, 8, 3], label='Neuron Inhibition')
                axs[layerIx].set_ylabel('potential')
              #  axs[layerIx].legend()

                if neuronId == numNeurons[layerIx,1]:
                    axs[layerIx].set_xlabel('time')

                if layerIx == 0 and neuronId == 0:
                    axs[layerIx].legend()


   # plt.show()


def visualizeSynapseMatrix(synapseMatrixHistory, title=""):

    exampleMatrix = synapseMatrixHistory[0]
    noPre, noPost = exampleMatrix.getNumberOfConnectedNeurons()

    ventricleCounts = np.zeros((noPre, noPost, len(synapseMatrixHistory)))
    criftTransmitterCounts = np.zeros((noPre, noPost, len(synapseMatrixHistory)))
    freeRezeptorCounts = np.zeros((noPre, noPost, len(synapseMatrixHistory)))
    acceptedTransmitterCounts = np.zeros((noPre, noPost, len(synapseMatrixHistory)))

    t = np.linspace(0, len(synapseMatrixHistory), len(synapseMatrixHistory))
    fig, axs = plt.subplots(noPre, noPost, sharex='all', sharey='all', figsize=(noPost * 2, noPre * 2))

    fig.suptitle(title)

    # Synapse Data
    for time in range(len(synapseMatrixHistory)):
        for preIx in range(noPre):
            for postIx in range(noPost):
                synapseMatrix = synapseMatrixHistory[time].synapseMatrix
                synapse = synapseMatrix[preIx, postIx].synapse
                ventricleCounts[preIx, postIx,time] = synapse[0]
                criftTransmitterCounts[preIx, postIx,time] = synapse[1]
                freeRezeptorCounts[preIx, postIx,time] = synapse[2]
         #       acceptedTransmitterCounts[preIx, postIx] = synapse[3]

    for preIx in range(noPre):
        for postIx in range(noPost):
            axs[preIx, postIx].set_title('Synapse ' + str(preIx) + ' to ' + str(postIx))  # 0 to 1')
            line1, = axs[preIx, postIx].plot(t, ventricleCounts[preIx, postIx], dashes=[2, 2, 10, 2],
                                             label='Ventricle Count')
            # line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
            line2, = axs[preIx, postIx].plot(t, criftTransmitterCounts[preIx, postIx], dashes=[6, 2],
                                             label='Clift Transmitter Count')
            line3, = axs[preIx, postIx].plot(t, freeRezeptorCounts[preIx, postIx], dashes=[6, 2, 4, 5],
                                             label='Free Rezeptor Count')
   #         line4, = axs[preIx, postIx].plot(t, acceptedTransmitterCounts[preIx, postIx], dashes=[8, 2, 2, 5],
    #                                         label='Accepted Transmitters Count')
            if preIx == 0 and postIx == 0:
                axs[preIx, postIx].legend()

            if postIx == noPost:
                axs[preIx, postIx].set_xlabel('time')
   # plt.show()

def visualizeNetwork(networkHistory, visualizeSynapses = True, visualizeNeurons = True):

    plt.style.use('bmh')
    #plt.style.use('Solarize_Light2')
    #plt.style.use('fivethirtyeight')
    #plt.style.use('dark_background')

    if visualizeSynapses:
        excitatoryMatrixHistory = np.full((networkHistory[0].numLayers, len(networkHistory)), L.SynapseMatrix)
        inhibitoryMatrixHistory = np.full((networkHistory[0].numLayers, len(networkHistory)), L.SynapseMatrix)
        lateralInhibitoryMatrixHistory = np.full((networkHistory[0].numLayers, len(networkHistory)), L.SynapseMatrix)

    if visualizeNeurons:
        allNeuronsHistory = []

    for time in range(len(networkHistory)):
        network = networkHistory[time]
        if visualizeNeurons:
            allNeuronsHistory.append(network.logNeurons())
        if visualizeSynapses:
            iterLayers = iter(network.layers)
            next(iterLayers)  # skipping the first layer
            for layerId, layer in enumerate(iterLayers):
                # TODO check which kind of layer is to be visualized
                excitatoryMatrixHistory[layerId,time] = layer.excitatoryMatrix
                lateralInhibitoryMatrixHistory[layerId,time] = layer.lateralInhibitoryMatrix

                if (isinstance(layer, L.DuoLateralInhibitoryLayer)):
                    inhibitoryMatrixHistory[layerId, time] = layer.inhibitoryMatrix


    exampleNetwork = networkHistory[0]
    if visualizeNeurons:
        allNeuronsHistory = np.array(allNeuronsHistory)
        visualizeNeuronPotentials(exampleNetwork, allNeuronsHistory)

    if visualizeSynapses:
        iterLayers = iter(exampleNetwork.layers)
        next(iterLayers)  # skipping the first layer
        for layerId, layer in enumerate(iterLayers):
            visualizeSynapseMatrix(excitatoryMatrixHistory[layerId], 'Layer ' + str(layerId) + ': Excitatory Synapses')
            visualizeSynapseMatrix(lateralInhibitoryMatrixHistory[layerId], 'Layer ' + str(layerId) + ': lateral Inhibitory Synapses')

            if (isinstance(layer, L.DuoLateralInhibitoryLayer)):
                visualizeSynapseMatrix(inhibitoryMatrixHistory[layerId],
                                       'Layer ' + str(layerId) + ': Inhibitory Synapses')

    plt.show()


# ----------------------------- STDP - visualisation ---------------------------

def visualiseLearningSignalHistory(networkHistory, calculations, timePerCalculation):
    exampleNetwork = networkHistory[0]
    exitoryLearningSignalsHistory = np.empty((exampleNetwork.numLayers, calculations, 3))
    inhibitoryLearningSignalsHistory = np.empty((exampleNetwork.numLayers, calculations, 3))

    excitatorySubsA = np.empty((exampleNetwork.numLayers, calculations, 1))
    excitatorySubsB = np.empty((exampleNetwork.numLayers, calculations, 1))
    excitatorySubsC = np.empty((exampleNetwork.numLayers, calculations, 1))
    effectiveExcitatoryAdjustment = np.empty((exampleNetwork.numLayers, calculations, 1))

    inhibSubsA = np.empty((exampleNetwork.numLayers, calculations, 1))
    inhibSubsB = np.empty((exampleNetwork.numLayers, calculations, 1))
    inhibSubsC = np.empty((exampleNetwork.numLayers, calculations, 1))
    effectiveInhibitoryAdjustment = np.empty((exampleNetwork.numLayers, calculations, 1))

    for c in range(0, calculations):
        # prepare visualisation of the learning signal
        cal = c + 1
        network = networkHistory[cal * (timePerCalculation - 1)]

        iterLayers = iter(network.layers)
        next(iterLayers)  # skipping the first layer
        for layerId, layer in enumerate(iterLayers):
            excitatorySynapseMatrix = layer.excitatoryMatrix.synapseMatrix
            excitatorySynapse = excitatorySynapseMatrix[0, 0].synapse
            exitoryLearningSignalsHistory[layerId, c] = excitatorySynapse[5]

            inhibitorySynapseMatrix = layer.inhibitoryMatrix.synapseMatrix
            inhibitorySynapse = inhibitorySynapseMatrix[0, 0].synapse
            inhibitoryLearningSignalsHistory[layerId, c] = inhibitorySynapse[5]

            if c == 0:
                excitatorySubsA[layerId, c] = exitoryLearningSignalsHistory[layerId, c, 0]
                excitatorySubsB[layerId, c] = exitoryLearningSignalsHistory[layerId, c, 1]
                excitatorySubsC[layerId, c] = exitoryLearningSignalsHistory[layerId, c, 2]

                inhibSubsA[layerId, c] = inhibitoryLearningSignalsHistory[layerId, c, 0]
                inhibSubsB[layerId, c] = inhibitoryLearningSignalsHistory[layerId, c, 1]
                inhibSubsC[layerId, c] = inhibitoryLearningSignalsHistory[layerId, c, 2]

            if c > 0:
                excitatorySubsA[layerId, c] = \
                    exitoryLearningSignalsHistory[layerId, c, 0] - exitoryLearningSignalsHistory[layerId, c-1, 0]
                excitatorySubsB[layerId, c] = \
                    exitoryLearningSignalsHistory[layerId, c, 1] - exitoryLearningSignalsHistory[layerId, c-1, 1]
                excitatorySubsC[layerId, c] = \
                    exitoryLearningSignalsHistory[layerId, c, 2] - exitoryLearningSignalsHistory[layerId, c-1, 2]

                inhibSubsA[layerId, c] = \
                    inhibitoryLearningSignalsHistory[layerId, c, 0] - inhibitoryLearningSignalsHistory[layerId, c-1, 0]
                inhibSubsB[layerId, c] = \
                    inhibitoryLearningSignalsHistory[layerId, c, 1] - inhibitoryLearningSignalsHistory[layerId, c-1, 1]
                inhibSubsC[layerId, c] = \
                    inhibitoryLearningSignalsHistory[layerId, c, 2] - inhibitoryLearningSignalsHistory[layerId, c-1, 2]

    effectiveExcitatoryAdjustment[layerId] = excitatorySubsC[layerId] - excitatorySubsB[layerId]
    effectiveInhibitoryAdjustment[layerId] = inhibSubsB[layerId] - inhibSubsC[layerId]

    iterLayers = iter(network.layers)
    next(iterLayers)  # skipping the first layer
    for layerId, layer in enumerate(iterLayers):
        cs = np.linspace((int(-calculations / 2)), int(calculations / 2), calculations)
        fig, axs = plt.subplots(3, 1, sharex='col', figsize=(3, 3))

        # ----
        axs[0].set_title('Excitatory Learning Signals')
        line1, = axs[0].plot(cs, excitatorySubsA[layerId], dashes=[2, 2, 10, 2], label='Substance A')
        line2, = axs[0].plot(cs, excitatorySubsB[layerId], dashes=[6, 2], label='Substance B - Weaken')
        line3, = axs[0].plot(cs, excitatorySubsC[layerId], dashes=[6, 2, 4, 2], label='Substance C - Enforce')
        line4, = axs[0].plot(cs, effectiveExcitatoryAdjustment[layerId], dashes=[3, 1, 5, 2], label='Effective Change')

        axs[0].set_ylabel('Intensity of the change') # in the information transmission effect')

        # set(leg1,'Interpreter','latex');
        # set(leg1,'FontSize',17);
        axs[0].legend()

        axs[1].set_title('Inhibitory Learning Signals')
        line5, = axs[1].plot(cs, inhibSubsA[layerId], label='Inhibitory Substance A')
        line5.set_dashes([8, 4, 10, 4])  # 2pt line, 2pt break, 10pt line, 2pt break
        line6, = axs[1].plot(cs, inhibSubsB[layerId], dashes=[6, 2, 2, 2], label='Inhibitory Substance B - Enforce')
        line7, = axs[1].plot(cs, inhibSubsC[layerId], dashes=[6, 4, 6, 2], label='Inhibitory Substance C - Weaken')
        line8, = axs[1].plot(cs, effectiveInhibitoryAdjustment[layerId], dashes=[4, 1, 6, 1, 4, 1],
                             label='Inhibitory Effective Change')
        axs[1].set_ylabel('Intensity of the change')
        axs[1].legend()

        axs[2].set_title('Learning Signals')
        line9, = axs[2].plot(cs, effectiveInhibitoryAdjustment[layerId], dashes=[4, 1, 6, 1, 4, 1],
                             label='Inhibitory Effective Change')
        line10, = axs[2].plot(cs, effectiveExcitatoryAdjustment[layerId], dashes=[3, 1, 5, 2], label='Effective Change')

        axs[2].set_ylabel('Intensity of the change')
        axs[2].set_xlabel('Time difference $\Delta(Spike_{Preneuron}, Spike_{Postneuron})$')
        axs[2].legend()

    plt.show()


# ----------------------------- Iris - visualisation ---------------------------
