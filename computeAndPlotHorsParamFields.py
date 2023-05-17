import numpy as np
import os, pickle

from BasicTools.Containers import UnstructuredMeshCreationTools as UMCT
from BasicTools.IO import XdmfWriter as XW
from BasicTools.Containers import UnstructuredMeshFieldOperations as UMFO
from BasicTools.FE.Fields import FEField as FF
from BasicTools.FE import FETools as FE
from utils import RenumberMeshForParametrization, FloaterMeshParametrization


"""
In this file, the prediction of scalar and field outputs for the out-of-distribution shapes ellipsoid and wedge
is illustrated, with plotting of the fields. Notice that  precomputation data is used, and the complete inference
workflow is provided and commented.
"""

nMonteCarloSamples = 10000

# Load the provided datasets
file = open("data/datasets.pkl", 'rb')
data = pickle.load(file)
file.close()

inMeshTrain = data["inMeshTrain"]
nTrain = len(data["inScalarsTrain"])

inMeshTestNewForm = data["inMeshTestNewForm"]
inScalarsTestNewForm = data["inScalarsTestNewForm"]
outScalarsTestNewForm = data["outScalarsTestNewForm"]
outFieldsTestNewForm = data["outFieldsTestNewForm"]

outFieldsNames = data["outFieldsNames"]
outScalarsNames = data["outScalarsNames"]

nOutFields = len(outFieldsNames)
nOutScalars = len(outScalarsNames)


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

scalerYScalars = trainingWorkflowData["scalerYScalars"]
rescaledYScalars = trainingWorkflowData["rescaledYScalars"]

fieldsRegressors = trainingWorkflowData["fieldsRegressors"]
scalarsRegressors = trainingWorkflowData["scalarsRegressors"]

correlationOperator1c = trainingWorkflowData["correlationOperator1c"]
correlationOperator2c = trainingWorkflowData["correlationOperator2c"]


# Clean previous plots
os.system('rm -rf predictHorsParamPlots')
os.system('mkdir predictHorsParamPlots')


# Compute the morphing of the median mesh
medianIndex = int(nTrain/2)
medianMesh = UMCT.CreateMeshOfTriangles(inMeshTrain[medianIndex]['points'].astype(np.float32), inMeshTrain[medianIndex]['triangles'])
medianMeshRenumb, renumb, nBoundary = RenumberMeshForParametrization(medianMesh, inPlace = False)
commonProjectionMesh, infos = FloaterMeshParametrization(medianMeshRenumb, nBoundary)
nNodeCommonMesh = commonProjectionMesh.GetNumberOfNodes()


#### Create field plots Xdmf files ####

plotOutFieldsNames = [n+"_predicted" for n in outFieldsNames] + \
                     [n+"_reference" for n in outFieldsNames] + \
                     [n+"_relative_error" for n in outFieldsNames] + \
                     [n+"_variance" for n in outFieldsNames] + \
                     [n+"_quantile_0.025" for n in outFieldsNames] + \
                     [n+"_quantile_0.975" for n in outFieldsNames]



nameCases = ["ellispoid", "wedge"]

for i in range(len(nameCases)):

    # Plot fields

    # Compute the morphing of the current mesh
    mesh = UMCT.CreateMeshOfTriangles(inMeshTestNewForm[i]['points'].astype(np.float32), inMeshTestNewForm[i]['triangles'])
    meshRenumb, renumb, nBoundary = RenumberMeshForParametrization(mesh, inPlace = False)
    meshParam, infos = FloaterMeshParametrization(meshRenumb, nBoundary)

    # Compute the two FE projection operators
    space, numberings, _, _ = FE.PrepareFEComputation(meshParam, numberOfComponents = 1)
    inputFEField = FF.FEField(name="edge",mesh=meshParam, space=space, numbering=numberings[0])
    operator, status = UMFO.GetFieldTransferOp(inputFEField, commonProjectionMesh.nodes, method = "Interp/Clamp", verbose=True)

    space, numberings, _, _ = FE.PrepareFEComputation(commonProjectionMesh, numberOfComponents = 1)
    inputFEField = FF.FEField(name="edge",mesh=commonProjectionMesh, space=space, numbering=numberings[0])
    invOperator, status = UMFO.GetFieldTransferOp(inputFEField, meshParam.nodes, method ="Interp/Clamp", verbose=True)

    # Compute the FE interpolation of the spatial coordinate field and output fields of interest
    projInCoordFields = operator.dot(mesh.nodes[renumb,:]).flatten()
    projOutFields = operator.dot(outFieldsTestNewForm[i][renumb,:])

    # Rescale the nongeometrical input scalars
    rescaledInScalar = scalerInScalar.transform(inScalarsTestNewForm[0,:].reshape(1,-1))[0,:]

    # Reduce the dimension of the precompputed projected spatial coordinate fields using the precomputed snapshots-POD
    reducedProjInCoordFields = np.dot(podProjInCoordFields, correlationOperator2c.dot(projInCoordFields.T)).T

    # Create the low-dimensional input by concatenating the two previous inputs
    X = np.hstack((reducedProjInCoordFields, rescaledInScalar)).reshape(1,-1)

    # Rescale the input
    rescaledX = scalerX.transform(X)

    predictedProjOutFields = np.empty((nNodeCommonMesh, nOutFields))
    predictedProjOutFieldsSamples = np.empty((nNodeCommonMesh, nOutFields, nMonteCarloSamples))
    for j in range(nOutFields):
        # Evaluate the field regressions (prediction and Monte Carlo draws)
        rescaledYFields = fieldsRegressors[j].predict(rescaledX)[0]
        rescaledYFieldsSamples = fieldsRegressors[j].posterior_samples_f(rescaledX.reshape(1,-1), size = nMonteCarloSamples)[0]

        # Inverse rescale the field outputs
        y = scalerYFields[j].inverse_transform(rescaledYFields)
        for k in range(nMonteCarloSamples):
            yTestSamples = scalerYFields[j].inverse_transform(rescaledYFieldsSamples[:,k].reshape(1,-1))

        # Decompress the field outputs using the pretrained snapshot-POD
        predictedProjOutFields[:,j] = np.dot(y, podProjOutFields[j])[0,:]
        for k in range(nMonteCarloSamples):
            yTestSamples = scalerYFields[j].inverse_transform(rescaledYFieldsSamples[:,k].reshape(1,-1))

    # Store the predictions
    invRenumb = np.argsort(renumb)
    predictedOutFields = invOperator.dot(predictedProjOutFields)[invRenumb,:]
    nNodes = predictedOutFields.shape[0]
    predictedOutFieldsSamples = np.empty((nNodes, nOutFields, nMonteCarloSamples))
    for k in range(nMonteCarloSamples):
        predictedOutFieldsSamples[:,:,k] = invOperator.dot(predictedProjOutFieldsSamples[:,:,k])[invRenumb,:]

    # Compute variance and quantiles from the Monte Carlo draws
    predictedOutFieldsVariance = np.var(predictedOutFieldsSamples, axis=2)
    predictedOutFieldsQuantile0_025 = np.quantile(predictedOutFieldsSamples, 0.025, axis=2)
    predictedOutFieldsQuantile0_975 = np.quantile(predictedOutFieldsSamples, 0.975, axis=2)

    predictedList = [predictedOutFields[:,j] for j in range(nOutFields)]
    referenceList = [outFieldsTestNewForm[i][:,j] for j in range(nOutFields)]

    # Compute relative errors
    maxes = [np.max(np.abs(outFieldsTestNewForm[i][:,j])) for j in range(nOutFields)]
    maxes = [m if m>0 else 1 for m in maxes]
    relativeErrorList = [np.abs(predictedOutFields[:,j]-outFieldsTestNewForm[i][:,j])/maxes[j] for j in range(nOutFields)]

    # Compute variance and quantiles
    varianceList = [predictedOutFieldsVariance[:,j] for j in range(nOutFields)]
    q0_025List = [predictedOutFieldsQuantile0_025[:,j] for j in range(nOutFields)]
    q0_975List = [predictedOutFieldsQuantile0_975[:,j] for j in range(nOutFields)]

    # Write Xdmf file
    pointFields = predictedList + referenceList + relativeErrorList + varianceList + q0_025List + q0_975List
    pointFields = [f.astype(np.float32) for f in pointFields]
    XW.WriteMeshToXdmf("predictHorsParamPlots/predictPlotsHorsParam_"+nameCases[i]+".xdmf", mesh,
        PointFields = pointFields, PointFieldsNames = plotOutFieldsNames)


    #############
    # Print scalars

    predictedOutScalars = np.empty(len(outScalarsNames))
    predictedProjOutScalarSamples = np.empty((nOutScalars, nMonteCarloSamples))

    for j in range(len(outScalarsNames)):
        # Evaluate the scalar regressions (prediction and Monte Carlo draws)
        rescaledYScalars = scalarsRegressors[j].predict(rescaledX)[0]
        rescaledYScalarsSamples = scalarsRegressors[j].posterior_samples_f(rescaledX.reshape(1,-1), size = nMonteCarloSamples)[0]

        # Inverse rescale the scalar outputs
        predictedOutScalars[j] = scalerYScalars[j].inverse_transform(rescaledYScalars.reshape(-1, 1))[:,0]
        for k in range(nMonteCarloSamples):
            predictedProjOutScalarSamples[j,k] = scalerYScalars[j].inverse_transform(rescaledYScalarsSamples[:,k].reshape(-1, 1))[:,0]

    # Compute variance and quantiles from the Monte Carlo draws
    predictedOutScalarsVariance = np.var(predictedProjOutScalarSamples, axis=1)
    predictedOutScalarsQuantile0_025 = np.quantile(predictedProjOutScalarSamples, 0.025, axis=1)
    predictedOutScalarsQuantile0_975 = np.quantile(predictedProjOutScalarSamples, 0.975, axis=1)

    print("Scalars for "+nameCases[i])
    for j in range(len(outScalarsNames)):
        print(outScalarsNames[j]+":")
        print("reference  =", outScalarsTestNewForm[i,j])
        print("prediction =", predictedOutScalars[j])
        print("relative error =", np.abs(predictedOutScalars[j]-outScalarsTestNewForm[i,j])/np.abs(outScalarsTestNewForm[i,j]))
        print("var     =", predictedOutScalarsVariance[j])
        print("q_0.025 =", predictedOutScalarsQuantile0_025[j])
        print("q_0.975 =", predictedOutScalarsQuantile0_975[j])
        print("---")

    print("===")
