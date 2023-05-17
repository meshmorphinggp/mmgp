import numpy as np
import pickle
from sklearn.metrics import r2_score
from BasicTools.Helpers import ProgressBar as PB

"""
In this file, the inference stage is computed for the scalars output,
for the data of the training and testing sets. Projected spatial coordinate
fields are exploited have been computed in a previous step and simply read here.

The inference stage contains a prediction for the values, variance and quantiles
0.025 and 0.975 using Monte Carlo draws.
"""

nMonteCarloSamples = 10000

# Load the provided datasets
file = open("data/datasets.pkl", 'rb')
data = pickle.load(file)
file.close()

inScalarsTrain = data["inScalarsTrain"]
outScalarsTrain = data["outScalarsTrain"]

inScalarsTest = data["inScalarsTest"]
outScalarsTest = data["outScalarsTest"]

outScalarsNames = data["outScalarsNames"]

nOutScalars = len(outScalarsNames)


# Load the pretreated data
file = open("data/pretreatedData.pkl", 'rb')
pretreatedData = pickle.load(file)
file.close()

projInCoordFieldsTrain = pretreatedData["projInCoordFieldsTrain"]
projInCoordFieldsTest = pretreatedData["projInCoordFieldsTest"]

nNodeCommonMesh = pretreatedData["projOutFieldsTest"].shape[1]


# Load the data constructing during the training workflow
file = open("data/trainingWorkflowData.pkl", 'rb')
trainingWorkflowData = pickle.load(file)
file.close()

podProjInCoordFields = trainingWorkflowData["podProjInCoordFields"]
scalerInScalar = trainingWorkflowData["scalerInScalar"]
scalerX = trainingWorkflowData["scalerX"]

scalerYScalars = trainingWorkflowData["scalerYScalars"]
rescaledYScalars = trainingWorkflowData["rescaledYScalars"]
scalarsRegressors = trainingWorkflowData["scalarsRegressors"]

correlationOperator2c = trainingWorkflowData["correlationOperator2c"]


#### Scalar prediction on the training set ####

nTrain = inScalarsTrain.shape[0]

# Reduce the dimension of the precompputed projected spatial coordinate fields using the precomputed snapshots- POD
reducedProjInCoordFields = np.dot(podProjInCoordFields, correlationOperator2c.dot(projInCoordFieldsTrain.T)).T

# Rescale the nongeometrical input scalars
rescaledInScalar = scalerInScalar.transform(inScalarsTrain)

# Create the low-dimensional input by concatenating the two previous inputs
XTrain = np.hstack((reducedProjInCoordFields, rescaledInScalar))

# Rescale the input
rescaledXTrain = scalerX.transform(XTrain)

predictedOutScalarsTrain = np.empty(outScalarsTrain.shape)
predictedOutScalarsTrainVariance = np.empty(outScalarsTrain.shape)
predictedOutScalarsTrainQuantile0_025 = np.empty(outScalarsTrain.shape)
predictedOutScalarsTrainQuantile0_975 = np.empty(outScalarsTrain.shape)

PB.printProgressBar(0, nTrain, prefix = '[Train] (Scalars) Predicting + UQ:', suffix = 'Complete', length = 50)
for i in range(nTrain):

    predictedProjOutScalarTrain = np.empty(nOutScalars)
    predictedProjOutScalarTrainSamples = np.empty((nOutScalars, nMonteCarloSamples))
    for j in range(nOutScalars):
        # Evaluate the scalar regressions (prediction and Monte Carlo draws)
        rescaledYScalarsTrain = scalarsRegressors[j].predict(rescaledXTrain[i,:].reshape(1,-1))[0]
        rescaledYScalarsSamples = scalarsRegressors[j].posterior_samples_f(rescaledXTrain[i,:].reshape(1,-1), size = nMonteCarloSamples)[0]

        # Inverse rescale the scalar outputs
        predictedOutScalarsTrain[i,j] = scalerYScalars[j].inverse_transform(rescaledYScalarsTrain.reshape(-1, 1))[:,0]
        for k in range(nMonteCarloSamples):
            predictedProjOutScalarTrainSamples[j,k] = scalerYScalars[j].inverse_transform(rescaledYScalarsSamples[:,k].reshape(-1, 1))[:,0]

    # Compute variance and quantiles from the Monte Carlo draws
    predictedOutScalarsTrainVariance[i,:] = np.var(predictedProjOutScalarTrainSamples, axis=1)
    predictedOutScalarsTrainQuantile0_025[i,:] = np.quantile(predictedProjOutScalarTrainSamples, 0.025, axis=1)
    predictedOutScalarsTrainQuantile0_975[i,:] = np.quantile(predictedProjOutScalarTrainSamples, 0.975, axis=1)

    PB.printProgressBar(i + 1, nTrain, prefix = '[Train] (Scalars) Predicting + UQ:', suffix = 'Complete', length = 50)

print("===")


#### Scalar prediction on the testing set ####

nTest = inScalarsTest.shape[0]

# reduce the dimension of the precompputed projected spatial coordinate fields using the precomputed snapshots-POD
reducedProjInCoordFields = np.dot(podProjInCoordFields, correlationOperator2c.dot(projInCoordFieldsTest.T)).T

# Rescale the nongeometrical input scalars
rescaledInScalar = scalerInScalar.transform(inScalarsTest)

# Create the low-dimensional input by concatenating the two previous inputs
XTest = np.hstack((reducedProjInCoordFields, rescaledInScalar))

# Rescale the input
rescaledXTest = scalerX.transform(XTest)

predictedOutScalarsTest = np.empty(outScalarsTest.shape)
predictedOutScalarsTestVariance = np.empty(outScalarsTest.shape)
predictedOutScalarsTestQuantile0_025 = np.empty(outScalarsTest.shape)
predictedOutScalarsTestQuantile0_975 = np.empty(outScalarsTest.shape)

PB.printProgressBar(0, nTest, prefix = '[Test] (Scalars) Predicting + UQ:', suffix = 'Complete', length = 50)
for i in range(nTest):

    predictedProjOutScalarTest = np.empty(nOutScalars)
    predictedProjOutScalarTestSamples = np.empty((nOutScalars, nMonteCarloSamples))
    for j in range(nOutScalars):
        # Evaluate the scalar regressions (prediction and Monte Carlo draws)
        rescaledYScalarsTest = scalarsRegressors[j].predict(rescaledXTest[i,:].reshape(1,-1))[0]
        rescaledYScalarsSamples = scalarsRegressors[j].posterior_samples_f(rescaledXTest[i,:].reshape(1,-1), size = nMonteCarloSamples)[0]

        # Inverse rescale the scalar outputs
        predictedOutScalarsTest[i,j] = scalerYScalars[j].inverse_transform(rescaledYScalarsTest.reshape(-1, 1))[:,0]
        for k in range(nMonteCarloSamples):
            predictedProjOutScalarTestSamples[j,k] = scalerYScalars[j].inverse_transform(rescaledYScalarsSamples[:,k].reshape(-1, 1))[:,0]

    # Compute variance and quantiles from the Monte Carlo draws
    predictedOutScalarsTestVariance[i,:] = np.var(predictedProjOutScalarTestSamples, axis=1)
    predictedOutScalarsTestQuantile0_025[i,:] = np.quantile(predictedProjOutScalarTestSamples, 0.025, axis=1)
    predictedOutScalarsTestQuantile0_975[i,:] = np.quantile(predictedProjOutScalarTestSamples, 0.975, axis=1)

    PB.printProgressBar(i + 1, nTest, prefix = '[Test] (Scalars) Predicting + UQ:', suffix = 'Complete', length = 50)


print("=== Q2 evaluations using "+str(nTest)+" testing samples ===")
for j in range(nOutScalars):
    print("Scalar output "+outScalarsNames[j]+" ; Q2 =", r2_score(outScalarsTest[:,j], predictedOutScalarsTest[:,j]))


# Save the predicted scalars data

predictedScalarsData = {}

predictedScalarsData["predictedOutScalarsTrain"] = predictedOutScalarsTrain
predictedScalarsData["predictedOutScalarsTrainVariance"] = predictedOutScalarsTrainVariance
predictedScalarsData["predictedOutScalarsTrainQuantile0_025"] = predictedOutScalarsTrainQuantile0_025
predictedScalarsData["predictedOutScalarsTrainQuantile0_975"] = predictedOutScalarsTrainQuantile0_975

predictedScalarsData["predictedOutScalarsTest"] = predictedOutScalarsTest
predictedScalarsData["predictedOutScalarsTestVariance"] = predictedOutScalarsTestVariance
predictedScalarsData["predictedOutScalarsTestQuantile0_025"] = predictedOutScalarsTestQuantile0_025
predictedScalarsData["predictedOutScalarsTestQuantile0_975"] = predictedOutScalarsTestQuantile0_975

with open("data/predictedScalarsData.pkl", 'wb') as file:
    pickle.dump(predictedScalarsData, file)
