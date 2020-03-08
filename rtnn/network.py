from . import Layers as L

import numpy as np

from copy import deepcopy

class Network():
    def __init__(self):
        self.layers = []
        self.numLayers = 0

    def addLayer(self, newLayer):
        if not self.layers:
            if not isinstance(newLayer, L.InputLayer):
                raise TypeError('The first layer of the network needs to be an Layers.InputLayer')
            else:
                self.layers.append(newLayer)
                self.numLayers += 1
        else: # list is not empty
            # TODO: check if the given argument is a layer
            nextInputLayer = self.layers[-1]
            preneurons = nextInputLayer.postneurons
            newLayer.build(preneurons)
            self.layers.append(newLayer)
            self.numLayers += 1

        return self

    # TODO loop could be run asynchronously in parallel
    # with n layers, this function could run in delayed-parallel after n initial computations.
    def step(self, excitatoryInputsT, inhibitoryInputsT, learning = True, layerBiasDict=None, ignorePreneurons = False):
        inputLayer = self.layers[0]
        inputLayer.step(excitatoryInputsT, inhibitoryInputsT)

        preneurons = inputLayer.preneurons

        iterLayers = iter(self.layers)
        next(iterLayers) #skipping the first layer
        for layer in iterLayers:
            if layerBiasDict:
                #print('Bias Dict exists')
                if layer in layerBiasDict:
                #    print(layer, ' in Bias Dict')
                    excitatoryBias, inhibitoryBias = layerBiasDict[layer]
                    layer.step(preneurons, learning, excitatoryBias, inhibitoryBias, ignorePreneurons)
                else:
                    layer.step(preneurons, learning)
            else:
                layer.step(preneurons, learning)
            preneurons = layer.postneurons

        outputLayer = preneurons

        return self, outputLayer

    def logNeurons(self):
        allNeurons = []
        for layerIx, layer in enumerate(self.layers):
            layerNeurons = []
            for neuronIx, neuron in enumerate(layer.postneurons):
                layerNeurons.append(deepcopy(neuron))
            layerNeurons = np.array(layerNeurons)
            allNeurons.append(layerNeurons)
        allNeurons = np.array(allNeurons)
        return allNeurons


    def logNetwork(self):
        '''
        Logs the whole network for a specific timestep
        '''
        return deepcopy(self)

