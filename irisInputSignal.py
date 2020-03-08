import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from rtnn import inputUtil as Iu

# -------------------------Initialize input series-------------------------------
data = pd.read_csv(
    '/home/mario/Dokumente/Transmitterbased Model/Modularized/iris.csv')

species_dummy = pd.get_dummies(data["species"])

assigned_data = data.copy()
assigned_data = pd.concat([data, species_dummy], axis=1)

target = ["setosa", "versicolor", "virginica"]
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

y = assigned_data[target].copy()
X = assigned_data[features].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)


def scaleDataToInterval(data, minI, maxI):
    maxD = np.max(data)
    minD = np.min(data)
    f = lambda x: scale(x, minI, maxI, minD, maxD)
    return f(data)


def scale(x, newMin, newMax, oldMin, oldMax):
    return (x - oldMin) * ((newMax - newMin) / (oldMax - oldMin)) + newMin

#min p for T = 1500: 0.005
#max p for T = 1500: 0.55

sepal_length_Current = scaleDataToInterval(X_train["sepal_length"], 0.008, 0.5)
sepal_width_Current = scaleDataToInterval(X_train["sepal_width"], 0.008, 0.5)
petal_length_Current = scaleDataToInterval(X_train["petal_length"], 0.008, 0.5)
petal_width_Current = scaleDataToInterval(X_train["petal_width"], 0.008, 0.5)

sepal_length_inverse_Current = scaleDataToInterval(X_train["sepal_length"], 0.5, 0.008)
sepal_width_inverse_Current = scaleDataToInterval(X_train["sepal_width"], 0.5, 0.008)
petal_length_inverse_Current = scaleDataToInterval(X_train["petal_length"], 0.5, 0.008)
petal_width_inverse_Current = scaleDataToInterval(X_train["petal_width"], 0.5, 0.008)

inputCurrentsPlan = np.array([sepal_length_Current, sepal_width_Current, petal_length_Current, petal_width_Current])
inverseInputCurrentPlan = np.array([sepal_length_inverse_Current, sepal_width_inverse_Current,
                                    petal_length_inverse_Current, petal_width_inverse_Current])


def makeTrainInputCurrent(id, duration):

    '''
    poisson Distributed Current for train set
    :param id: index of id to generate spike train for
    :param duration: of the generated spike train
    :return: spike train for 4 positive and 4 inverse spike trains. One for each feature.
    '''

    p0, p1, p2, p3 = inputCurrentsPlan[:, id]
    ip0, ip1, ip2, ip3 = inverseInputCurrentPlan[:, id]
    fp = [p0, p1, p2, p3, ip0, ip1, ip2, ip3]
    preStartDuration = 350
    postStartDuration = 550
    smallCurrent = Iu.poissonDistributedSpikeTrain(duration - (preStartDuration + postStartDuration),
                                                   numberNeuronsInLayer=8, firingProbabilities=fp)
    current = np.concatenate([Iu.noSpike(preStartDuration, 8), smallCurrent, Iu.noSpike(postStartDuration, 8)])
    return current

'''
def noSpike(duration, noInputNeurons=3):
    return np.zeros(noInputNeurons * duration).reshape(duration, noInputNeurons)

def makeInputCurrent(id):
    input0, input1, input2, input3 = inputCurrentsPlan[:, id]
    smallCurrent = np.concatenate([np.array([[input0, input1, input2, input3]]), noSpike(1, 4)], axis=0)
    current = np.concatenate([noSpike(300, 4), np.tile(smallCurrent, (500, 1)), noSpike(500, 4)])
    return current
'''

spikeNeuron0 = Iu.spikeNeuron(0, noInputNeurons=3) #np.array([[20000, 0, 0]])
spikeNeuron1 = Iu.spikeNeuron(1, noInputNeurons=3) #np.array([[0, 20000, 0]])
spikeNeuron2 = Iu.spikeNeuron(2, noInputNeurons=3) #np.array([[0, 0, 20000]])


def makeOutputCurrent(id):
    null = y_train.iloc[id, 0]
    eins = y_train.iloc[id, 1]
    zwei = y_train.iloc[id, 2]
    current = Iu.noSpike(1600, 3)
    print('TargetClass: ', null, eins, zwei)
    if null == 1:
        smallCurrent = np.concatenate([spikeNeuron0, Iu.noSpike(25, 3)], axis=0)
        current = np.concatenate([Iu.noSpike(400, 3), np.tile(smallCurrent, (25, 1)), Iu.noSpike(650, 3)])
    if eins == 1:
        smallCurrent = np.concatenate([spikeNeuron1, Iu.noSpike(25, 3)], axis=0)
        current = np.concatenate([Iu.noSpike(400, 3), np.tile(smallCurrent, (25, 1)), Iu.noSpike(650, 3)])
    if zwei == 1:
        smallCurrent = np.concatenate([spikeNeuron2, Iu.noSpike(25, 3)], axis=0)
        current = np.concatenate([Iu.noSpike(400, 3), np.tile(smallCurrent, (25, 1)), Iu.noSpike(650, 3)])
    return current


sepal_length_TestCurrent = scaleDataToInterval(X_test["sepal_length"], 0.008, 0.5)
sepal_width_TestCurrent = scaleDataToInterval(X_test["sepal_width"], 0.008, 0.5)
petal_length_TestCurrent = scaleDataToInterval(X_test["petal_length"], 0.008, 0.5)
petal_width_TestCurrent = scaleDataToInterval(X_test["petal_width"], 0.008, 0.5)

sepal_length_inverse_TestCurrent = scaleDataToInterval(X_test["sepal_length"], 0.5, 0.008)
sepal_width_inverse_TestCurrent = scaleDataToInterval(X_test["sepal_width"], 0.5, 0.008)
petal_length_inverse_TestCurrent = scaleDataToInterval(X_test["petal_length"], 0.5, 0.008)
petal_width_inverse_TestCurrent = scaleDataToInterval(X_test["petal_width"], 0.5, 0.008)

inputTestCurrentsPlan = np.array(
    [sepal_length_TestCurrent, sepal_width_TestCurrent, petal_length_TestCurrent, petal_width_TestCurrent])

inputTestInverseCurrentsPlan = np.array([sepal_length_inverse_TestCurrent, sepal_width_inverse_TestCurrent,
                                         petal_length_inverse_TestCurrent, petal_width_inverse_TestCurrent])


def makeTestInputCurrent(id, duration):
    '''
    poisson Distributed Current for test set
    :param id: index of id to generate spike train for
    :param duration: of the generated spike train
    :return: spike train for 4 positive and 4 inverse spike trains. One for each feature.
    '''

    null = y_test.iloc[id, 0]
    eins = y_test.iloc[id, 1]
    zwei = y_test.iloc[id, 2]

    print('TargetClass: ', null, eins, zwei)

    p0, p1, p2, p3 = inputTestCurrentsPlan[:, id]
    ip0, ip1, ip2, ip3 = inputTestInverseCurrentsPlan[:, id]
    fp = [p0, p1, p2, p3, ip0, ip1, ip2, ip3]
    preStartDuration = 350
    postStartDuration = 550
    smallCurrent = Iu.poissonDistributedSpikeTrain(duration - (preStartDuration + postStartDuration),
                                                   numberNeuronsInLayer=8, firingProbabilities=fp)
    current = np.concatenate([Iu.noSpike(preStartDuration, 8), smallCurrent, Iu.noSpike(postStartDuration, 8)])
    return current


'''
def makeInputTestCurrent(id):
    null = y_test.iloc[id, 0]
    eins = y_test.iloc[id, 1]
    zwei = y_test.iloc[id, 2]

    print('TargetClass: ', null, eins, zwei)

    input0, input1, input2, input3 = inputTestCurrentsPlan[:, id]
    smallCurrent = np.concatenate([np.array([[input0, input1, input2, input3]]), noSpike(1, 4)], axis=0)
    current = np.concatenate([noSpike(300, 4), np.tile(smallCurrent, (500, 1)), noSpike(500, 4)])
    return current
'''

