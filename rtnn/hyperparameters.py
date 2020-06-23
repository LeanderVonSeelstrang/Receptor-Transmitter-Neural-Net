

# synapse matrix hyperparameters
# for Layer.py

EXCITATORY_RECEPTOR_VARIANCE = 12
EXCITATORY_RECEPTOR_COUNT = 1200
EXCITATORY_TRANSMITTER_VARIANCE = 9
EXCITATORY_TRANSMITTER_COUNT = 5000

LATERAL_INHIBITORY_RECEPTOR_VARIANCE = 7
LATERAL_INHIBITORY_RECEPTOR_COUNT = 1830
LATERAL_INHIBITORY_TRANSMITTER_VARIANCE = 16
LATERAL_INHIBITORY_TRANSMITTER_COUNT = 3510

INHIBITORY_RECEPTOR_VARIANCE = 14
INHIBITORY_RECEPTOR_COUNT = 1200
INHIBITORY_TRANSMITTER_VARIANCE = 8
INHIBITORY_TRANSMITTER_COUNT = 3500


# neuron hyperparameters
SPIKING_THRESHHOLD = 150
ACTION_POTENTIAL_VALUE = 600
REFRACTORY_TIME = 9

EXPECTED_NEURON_LEAK = 3
VARIANCE_LEAK = 1

EXPECTED_INHIBITION_LEAK = 2
VARIANCE_INHIBITION_LEAK = 2

EXPECTED_MAXIMAL_INHIBITION = SPIKING_THRESHHOLD * 2
VARIANCE_MAXIMAL_INHIBITION = 2


# synapse hyperparameters

MAX_CLIFT_TRANSMITTER_COUNT = 60000 #500000 #130000

COLLISION_SPEED = 0.00001386294361119890618834464242916353136151
RECEPTOR_BUILD_SPEED = 0.00897309
TRANSMITTER_BUILD_SPEED = 0.000002351386294361119890618834464242916353136151
EXPONENTIAL_CLIFT_LEAK_SPEED = 0.004

TIME_DIFFERENCE_SENSITIVITY_LEARNING_SIGNAL = 0.00005138629

RELEASE_PERCENTAGE = 0.1

LINEAR_CLIFT_LEAK = 3

