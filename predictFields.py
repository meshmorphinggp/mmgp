import numpy as np
import pickle
from sklearn.metrics import r2_score
from BasicTools.Helpers import ProgressBar as PB

"""
In this file, the inference stage is computed for the fields output,
for the data of the training and testing sets. Projected spatial coordinate
fields are exploited have been computed in a previous step and simply read here.

The inference stage contains a prediction for the values, variance and quantiles
0.025 and 0.975 using Monte Carlo draws.
"""

nMonteCarloSamples = 10000
nPlots = 2 # first samples of the training and testing sets to be plotted (set to 200 for computing Q2 en testing set)

# Load the provided datasets
file = open("data/datasets.pkl", 'rb')
data = pickle.load(file)
file.close()

inScalarsTrain = data["inScalarsTrain"]
outFieldsTrain = data["outFieldsTrain"]

inScalarsTest = data["inScalarsTest"]
outFieldsTest = data["outFieldsTest"]

outFieldsNames = data["outFieldsNames"]

nOutFields = len(outFieldsNames)


# Load the pretreated data
file = open("data/pretreatedData.pkl", 'rb')
pretreatedData = pickle.load(file)
file.close()

invProjOperatorsTrain = pretreatedData["invProjOperatorsTrain"]
projInCoordFieldsTrain = pretreatedData["projInCoordFieldsTrain"]
renumberingTrain = pretreatedData["renumberingTrain"]

invProjOperatorsTest = pretreatedData["invProjOperatorsTest"]
projInCoordFieldsTest = pretreatedData["projInCoordFieldsTest"]
renumberingTest = pretreatedData["renumberingTest"]

nNodeCommonMesh = pretreatedData["projOutFieldsTest"].shape[1]


# Load the data constructing during the training workflow
file = open("data/trainingWorkflowData.pkl", 'rb')
trainingWorkflowData = pickle.load(file)
file.close()

podProjInCoordFields = trainingWorkflowData["podProjInCoordFields"]
scalerInScalar = trainingWorkflowData["scalerInScalar"]
scalerX = trainingWorkflowData["scalerX"]

podProjOutFields = trainingWorkflowData["podProjOutFields"]
scalerYFields = trainingWorkflowData["scalerYFields"]
rescaledYFields = trainingWorkflowData["rescaledYFields"]

fieldsRegressors = trainingWorkflowData["fieldsRegressors"]

correlationOperator2c = trainingWorkflowData["correlationOperator2c"]


#### Field prediction on the training set ####

nTrain = nPlots

# Reduce the dimension of the precompputed projected spatial coordinate fields using the precomputed snapshots-POD
reducedProjInCoordFields = np.dot(podProjInCoordFields, correlationOperator2c.dot(projInCoordFieldsTrain.T)).T

# Rescale the nongeometrical input scalars
rescaledInScalar = scalerInScalar.transform(inScalarsTrain)

# Create the low-dimensional input by concatenating the two previous inputs
XTrain = np.hstack((reducedProjInCoordFields, rescaledInScalar))

# Rescale the input
rescaledXTrain = scalerX.transform(XTrain)


predictedOutFieldsTrain = []
predictedOutFieldsTrainVariance = []
predictedOutFieldsTrainQuantile0_025 = []
predictedOutFieldsTrainQuantile0_975 = []

PB.printProgressBar(0, nTrain, prefix = '[Train] (Fields) Predicting + UQ:', suffix = 'Complete', length = 50)
for i in range(nTrain):

    invRenumb = np.argsort(renumberingTrain[i])
    predictedProjOutFieldsTrain = np.empty((nNodeCommonMesh, nOutFields))
    predictedProjOutFieldsTrainSamples = np.empty((nNodeCommonMesh, nOutFields, nMonteCarloSamples))

    for j in range(nOutFields):
        # Evaluate the field regressions (prediction and Monte Carlo draws)
        rescaledYFieldsTrain = fieldsRegressors[j].predict(rescaledXTrain[i,:].reshape(1,-1))[0]
        rescaledYFieldsTrainSamples = fieldsRegressors[j].posterior_samples_f(rescaledXTrain[i,:].reshape(1,-1), size = nMonteCarloSamples)[0]

        # Inverse rescale the field outputs
        yTrain = scalerYFields[j].inverse_transform(rescaledYFieldsTrain.reshape(1,-1))
        for k in range(nMonteCarloSamples):
            yTrainSamples = scalerYFields[j].inverse_transform(rescaledYFieldsTrainSamples[:,k].reshape(1,-1))

        # Decompress the field outputs using the pretrained snapshot-POD
        predictedProjOutFieldsTrain[:,j] = np.dot(yTrain, podProjOutFields[j])
        for k in range(nMonteCarloSamples):
            predictedProjOutFieldsTrainSamples[:,j,k] = np.dot(yTrainSamples, podProjOutFields[j])[0,:]

    # Store the predictions
    predictedOutFieldsTrain.append(invProjOperatorsTrain[i].dot(predictedProjOutFieldsTrain)[invRenumb,:])
    nNodes = predictedOutFieldsTrain[i].shape[0]
    predictedOutFieldsTrainSamples = np.empty((nNodes, nOutFields, nMonteCarloSamples))
    for k in range(nMonteCarloSamples):
        predictedOutFieldsTrainSamples[:,:,k] = invProjOperatorsTrain[i].dot(predictedProjOutFieldsTrainSamples[:,:,k])[invRenumb,:]

    # Compute variance and quantiles from the Monte Carlo draws
    predictedOutFieldsTrainVariance.append(np.var(predictedOutFieldsTrainSamples, axis=2))
    predictedOutFieldsTrainQuantile0_025.append(np.quantile(predictedOutFieldsTrainSamples, 0.025, axis=2))
    predictedOutFieldsTrainQuantile0_975.append(np.quantile(predictedOutFieldsTrainSamples, 0.975, axis=2))

    PB.printProgressBar(i + 1, nTrain, prefix = '[Train] (Fields) Predicting + UQ:', suffix = 'Complete', length = 50)


#### Field prediction on the testing set ####

nTest = nPlots

# Reduce the dimension of the precompputed projected spatial coordinate fields using the precomputed snapshots-POD
reducedProjInCoordFields = np.dot(podProjInCoordFields, correlationOperator2c.dot(projInCoordFieldsTest.T)).T

# Rescale the nongeometrical input scalars
rescaledInScalar = scalerInScalar.transform(inScalarsTest)

# Create the low-dimensional input by concatenating the two previous inputs
XTest = np.hstack((reducedProjInCoordFields, rescaledInScalar))

# Rescale the input
rescaledXTest = scalerX.transform(XTest)


predictedOutFieldsTest = []
predictedOutFieldsTestVariance = []
predictedOutFieldsTestQuantile0_025 = []
predictedOutFieldsTestQuantile0_975 = []

PB.printProgressBar(0, nTest, prefix = '[Test] (Fields) Predicting + UQ:', suffix = 'Complete', length = 50)
for i in range(nTest):

    predictedProjOutFieldsTest = np.empty((nNodeCommonMesh, nOutFields))
    predictedProjOutFieldsTestSamples = np.empty((nNodeCommonMesh, nOutFields, nMonteCarloSamples))

    for j in range(nOutFields):
        # Evaluate the field regressions (prediction and Monte Carlo draws)
        rescaledYFieldsTest = fieldsRegressors[j].predict(rescaledXTest[i,:].reshape(1,-1))[0]
        rescaledYFieldsTestSamples = fieldsRegressors[j].posterior_samples_f(rescaledXTest[i,:].reshape(1,-1), size = nMonteCarloSamples)[0]

        # Inverse rescale the field outputs
        yTest = scalerYFields[j].inverse_transform(rescaledYFieldsTest.reshape(1,-1))
        for k in range(nMonteCarloSamples):
            yTestSamples = scalerYFields[j].inverse_transform(rescaledYFieldsTestSamples[:,k].reshape(1,-1))

        # Decompress the field outputs using the pretrained snapshot-POD
        predictedProjOutFieldsTest[:,j] = np.dot(yTest, podProjOutFields[j])
        for k in range(nMonteCarloSamples):
            predictedProjOutFieldsTestSamples[:,j,k] = np.dot(yTestSamples, podProjOutFields[j])[0,:]

    # Store the predictions
    invRenumb = np.argsort(renumberingTest[i])
    predictedOutFieldsTest.append(invProjOperatorsTest[i].dot(predictedProjOutFieldsTest)[invRenumb,:])
    nNodes = predictedOutFieldsTest[i].shape[0]
    predictedOutFieldsTestSamples = np.empty((nNodes, nOutFields, nMonteCarloSamples))
    for k in range(nMonteCarloSamples):
        predictedOutFieldsTestSamples[:,:,k] = invProjOperatorsTest[i].dot(predictedProjOutFieldsTestSamples[:,:,k])[invRenumb,:]

    # Compute variance and quantiles from the Monte Carlo draws
    predictedOutFieldsTestVariance.append(np.var(predictedOutFieldsTestSamples, axis=2))
    predictedOutFieldsTestQuantile0_025.append(np.quantile(predictedOutFieldsTestSamples, 0.025, axis=2))
    predictedOutFieldsTestQuantile0_975.append(np.quantile(predictedOutFieldsTestSamples, 0.975, axis=2))

    PB.printProgressBar(i + 1, nTest, prefix = '[Test] (Fields) Predicting + UQ:', suffix = 'Complete', length = 50)

print("=== Q2 evaluations using "+str(nTest)+" testing samples ===")
for j in range(nOutFields):
    predictVect = 0
    exactVect = 0
    for i in range(nTest):
        predictVect = np.hstack((predictVect, predictedOutFieldsTest[i][:,j]))
        exactVect = np.hstack((exactVect, outFieldsTest[i][:,j]))
    print("Field output "+outFieldsNames[j]+" ; Q2 =", r2_score(predictVect, exactVect))
print("===")


# Save the predicted fields data

predictedFieldsData = {}

predictedFieldsData["predictedOutFieldsTrain"] = predictedOutFieldsTrain
predictedFieldsData["predictedOutFieldsTrainVariance"] = predictedOutFieldsTrainVariance
predictedFieldsData["predictedOutFieldsTrainQuantile0_025"] = predictedOutFieldsTrainQuantile0_025
predictedFieldsData["predictedOutFieldsTrainQuantile0_975"] = predictedOutFieldsTrainQuantile0_975

predictedFieldsData["predictedOutFieldsTest"] = predictedOutFieldsTest
predictedFieldsData["predictedOutFieldsTestVariance"] = predictedOutFieldsTestVariance
predictedFieldsData["predictedOutFieldsTestQuantile0_025"] = predictedOutFieldsTestQuantile0_025
predictedFieldsData["predictedOutFieldsTestQuantile0_975"] = predictedOutFieldsTestQuantile0_975

with open("data/predictedFieldsData.pkl", 'wb') as file:
    pickle.dump(predictedFieldsData, file)