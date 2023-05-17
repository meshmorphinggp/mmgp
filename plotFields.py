import numpy as np
import os, pickle
from BasicTools.Containers import UnstructuredMeshCreationTools as UMCT
from BasicTools.IO import XdmfWriter as XW

"""
In this file, the code for plotting the MMGP output field predictions, variance,
0.025 and 0.975 quantiles, references and relative errors is given.

No new new computation is done, the data is only read and plotted in the Xdmf format,
natively readable by Paraview.
"""

nPlots = 2 # first samples of the training and testing sets to be plotted

# Clean previous plots
os.system('rm -rf predictFieldsPlots')
os.system('mkdir predictFieldsPlots')

# Load the provided datasets
file = open("data/datasets.pkl", 'rb')
data = pickle.load(file)
file.close()

inMeshTrain = data["inMeshTrain"]
inMeshTest = data["inMeshTest"]

outFieldsTrain = data["outFieldsTrain"]
outFieldsTest = data["outFieldsTest"]
outFieldsNames = data["outFieldsNames"]


# Load the predicted fields data
file = open("data/predictedFieldsData.pkl", 'rb')
dataTest = pickle.load(file)
file.close()

predictedOutFieldsTrain = dataTest["predictedOutFieldsTrain"]
predictedOutFieldsTrainVariance = dataTest["predictedOutFieldsTrainVariance"]
predictedOutFieldsTrainQuantile0_025 = dataTest["predictedOutFieldsTrainQuantile0_025"]
predictedOutFieldsTrainQuantile0_975 = dataTest["predictedOutFieldsTrainQuantile0_975"]

predictedOutFieldsTest = dataTest["predictedOutFieldsTest"]
predictedOutFieldsTestVariance = dataTest["predictedOutFieldsTestVariance"]
predictedOutFieldsTestQuantile0_025 = dataTest["predictedOutFieldsTestQuantile0_025"]
predictedOutFieldsTestQuantile0_975 = dataTest["predictedOutFieldsTestQuantile0_975"]


#### Create field plots Xdmf files ####

plotOutFieldsNames = [n+"_predicted" for n in outFieldsNames] + \
                     [n+"_reference" for n in outFieldsNames] + \
                     [n+"_relative_error" for n in outFieldsNames] + \
                     [n+"_variance" for n in outFieldsNames] + \
                     [n+"_quantile_0.025" for n in outFieldsNames] + \
                     [n+"_quantile_0.975" for n in outFieldsNames]


nOutFields = len(outFieldsNames)

#### Training set ####
print("===")
for i in range(nPlots):

    print("plotting training configuration", i)
    mesh = UMCT.CreateMeshOfTriangles(inMeshTrain[i]['points'].astype(np.float32), inMeshTrain[i]['triangles'])

    predictedList = [predictedOutFieldsTrain[i][:,j] for j in range(nOutFields)]
    referenceList = [outFieldsTrain[i][:,j] for j in range(nOutFields)]

    # Compute relative errors
    maxes = [np.max(np.abs(outFieldsTrain[i][:,j])) for j in range(nOutFields)]
    maxes = [m if m>0 else 1 for m in maxes]
    relativeErrorList = [np.abs(predictedOutFieldsTrain[i][:,j]-outFieldsTrain[i][:,j])/maxes[j] for j in range(nOutFields)]

    # Compute variance and quantiles
    varianceList = [predictedOutFieldsTrainVariance[i][:,j] for j in range(nOutFields)]
    q0_025List = [predictedOutFieldsTrainQuantile0_025[i][:,j] for j in range(nOutFields)]
    q0_975List = [predictedOutFieldsTrainQuantile0_975[i][:,j] for j in range(nOutFields)]

    # Write Xdmf file
    pointFields = predictedList + referenceList + relativeErrorList + varianceList + q0_025List + q0_975List
    pointFields = [f.astype(np.float32) for f in pointFields]
    XW.WriteMeshToXdmf("predictFieldsPlots/prediction_train_"+str(i)+".xdmf", mesh,
        PointFields = pointFields, PointFieldsNames = plotOutFieldsNames)


#### Training set ####
print("===")
for i in range(nPlots):

    print("plotting testing configuration", i)
    mesh = UMCT.CreateMeshOfTriangles(inMeshTest[i]['points'].astype(np.float32), inMeshTest[i]['triangles'])

    predictedList = [predictedOutFieldsTest[i][:,j] for j in range(nOutFields)]
    referenceList = [outFieldsTest[i][:,j] for j in range(nOutFields)]
    maxes = [np.max(np.abs(outFieldsTest[i][:,j])) for j in range(nOutFields)]
    maxes = [m if m>0 else 1 for m in maxes]
    relativeErrorList = [np.abs(predictedOutFieldsTest[i][:,j]-outFieldsTest[i][:,j])/maxes[j] for j in range(nOutFields)]

    varianceList = [predictedOutFieldsTestVariance[i][:,j] for j in range(nOutFields)]
    q0_025List = [predictedOutFieldsTestQuantile0_025[i][:,j] for j in range(nOutFields)]
    q0_975List = [predictedOutFieldsTestQuantile0_975[i][:,j] for j in range(nOutFields)]

    pointFields = predictedList + referenceList + relativeErrorList + varianceList + q0_025List + q0_975List
    pointFields = [f.astype(np.float32) for f in pointFields]

    XW.WriteMeshToXdmf("predictFieldsPlots/prediction_test_"+str(i)+".xdmf", mesh,
        PointFields = pointFields, PointFieldsNames = plotOutFieldsNames)


