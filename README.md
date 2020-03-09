# Chemical Neural Network

Information transfer in biology occurs via chemical transmitters, which are released with a preneuronal action potential into a synaptic cleft. The emitted transmitters are then taken up into the cell by associated receptors, where they have further effects.


## Neuron Model

This project uses a 'Leaky-Integrate and Fire' neuron model. Each neuron accumulates excitatory or inhibitory potential. If the excitatory potential crosses a threshold the neuron will fire and a bunch of transmitters will be released into the synapses.


## Transmitter Receptor based Synapse

The interface between two neurons, the synapses, are modelled via various differential equations that indicate the change in state given to the current state. 

A population model is used for the build-up rate of transmitter substances in presynaptic vesicles. 

For the coupling quantity, i.e. how many transmitter substances are taken up into the cell because they collide with receptors, a chemical reaction equation of second order is used, which describes the reaction rate of a reaction A + B -> C. 

To model how many occupied receptors are released per time step, an equation is used that models how the temperature of an object adapts to the outside temperature.


## Requirements

numpy
matplotlib

