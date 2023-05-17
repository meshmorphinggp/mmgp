import numpy as np
import GPy
import pickle
import time
from sklearn.preprocessing import StandardScaler
from BasicTools.FE import FETools as FT
from BasicTools.Containers import UnstructuredMeshCreationTools as UMCT
from utils import snapshotsPOD_fit_transform, RenumberMeshForParametrization, FloaterMeshParametrization

"""
In this file, the GPs are trained independently on low-dimensional pretreated data.
Rescalings are done in this file as well.
"""

# Load the provided datasets
file = open("data/datasets.pkl", 'rb')
data = pickle.load(file)
file.close()

inMeshTrain = data["inMeshTrain"]

inScalarsTrain = data["inScalarsTrain"]
outScalarsTrain = data["outScalarsTrain"]

outFieldsNames = data["outFieldsNames"]
outScalarsNames = data["outScalarsNames"]


# Load the pretreated data
file = open("data/pretreatedData.pkl", 'rb')
pretreatedData = pickle.load(file)
file.close()

projInCoordFieldsTrain = pretreatedData["projInCoordFieldsTrain"]
projOutFieldsTrain = pretreatedData["projOutFieldsTrain"]


nPODModesIn = 8
nPODModesOut = 8
nTrain = len(inScalarsTrain)



start = time.time()

# Compute the morphing of the median mesh
medianIndex = int(nTrain/2)
medianMesh = UMCT.CreateMeshOfTriangles(inMeshTrain[medianIndex]['points'].astype(np.float32), inMeshTrain[medianIndex]['triangles'])
medianMeshRenumb, renumb, nBoundary = RenumberMeshForParametrization(medianMesh, inPlace = False)
commonProjectionMesh, infos = FloaterMeshParametrization(medianMeshRenumb, nBoundary)

# Compute correlation operators needed for the snapshot-POD dimensionality reduction
correlationOperator1c = FT.ComputeL2ScalarProducMatrix(commonProjectionMesh, numberOfComponents = 1)
correlationOperator2c = FT.ComputeL2ScalarProducMatrix(commonProjectionMesh, numberOfComponents = 2)

# Construct the snapshot-POD dimensionality reduction for the projection of the coordinate fields onto the common mesh
podProjInCoordFields, reducedProjInCoordFields = snapshotsPOD_fit_transform(projInCoordFieldsTrain, correlationOperator2c, nPODModesIn)

# Construct a rescaling for the nongeometrical input scalars
scalerInScalar = StandardScaler()
rescaledInScalar = scalerInScalar.fit_transform(inScalarsTrain)

# Create the ML low-dimensional input by concatenating the two previous inputs
X = np.hstack((reducedProjInCoordFields, rescaledInScalar))

# Construct a rescaling for the input
scalerX = StandardScaler()
rescaledX = scalerX.fit_transform(X)

# Construct the snapshot-POD dimensionality reduction for the projection of each output field of interest onto the common mesh
podProjOutFields = []
scalerYFields = []
rescaledYFields = []
for i in range(projOutFieldsTrain.shape[2]):

    podTemp, yTemp = snapshotsPOD_fit_transform(projOutFieldsTrain[:,:,i], correlationOperator1c, nPODModesOut)

    podProjOutFields.append(podTemp)
    scalerYFields.append(StandardScaler())
    rescaledYFields.append(scalerYFields[i].fit_transform(yTemp))


# Declare the options for GPy (for the GPs)
options = {"kernel":"Matern52",
           "optim":"bfgs",
           "num_restarts":10,
           "max_iters":1000,
           "anisotrope":True}


# Train the GPs for the field outputs
print("---")
print("training fields")
print("---")
fieldsRegressors = []
for i in range(len(outFieldsNames)):

    print("training regressor for field "+outFieldsNames[i])
    print("---")

    kernalClass = getattr(GPy.kern, options["kernel"])
    k = kernalClass(input_dim=X.shape[1], ARD = options["anisotrope"])

    fieldsRegressors.append(GPy.models.GPRegression(rescaledX, rescaledYFields[i], k))
    fieldsRegressors[i].optimize_restarts(optimizer = options["optim"], max_iters = options["max_iters"], num_restarts = options["num_restarts"])

    print(fieldsRegressors[i])
    print("===")


# Train the GPs for the scalar outputs
scalerYScalars = []
rescaledYScalars = []
for i in range(outScalarsTrain.shape[1]):

    scalerYScalars.append(StandardScaler())
    rescaledYScalars.append(scalerYScalars[i].fit_transform(outScalarsTrain[:,i].reshape(-1, 1)))

print("---")
print("training scalars")
print("---")
scalarsRegressors = []
for i in range(len(outScalarsNames)):

    print("training regressor for scalar "+outScalarsNames[i])
    print("---")

    kernalClass = getattr(GPy.kern, options["kernel"])
    k = kernalClass(input_dim=X.shape[1], ARD = options["anisotrope"])

    scalarsRegressors.append(GPy.models.GPRegression(rescaledX, rescaledYScalars[i], k))
    scalarsRegressors[i].optimize_restarts(optimizer = options["optim"], max_iters = options["max_iters"], num_restarts = options["num_restarts"])

    print(scalarsRegressors[i])
    print("===")

print("duration for complete training =", time.time() - start)


# Save the data constructing during the training workflow

trainingWorkflowData = {}

trainingWorkflowData["podProjInCoordFields"] = podProjInCoordFields
trainingWorkflowData["scalerInScalar"] = scalerInScalar
trainingWorkflowData["scalerX"] = scalerX

trainingWorkflowData["podProjOutFields"] = podProjOutFields
trainingWorkflowData["scalerYFields"] = scalerYFields
trainingWorkflowData["rescaledYFields"] = rescaledYFields

trainingWorkflowData["scalerYScalars"] = scalerYScalars
trainingWorkflowData["rescaledYScalars"] = rescaledYScalars

trainingWorkflowData["fieldsRegressors"] = fieldsRegressors
trainingWorkflowData["scalarsRegressors"] = scalarsRegressors

trainingWorkflowData["correlationOperator1c"] = correlationOperator1c
trainingWorkflowData["correlationOperator2c"] = correlationOperator2c

with open("data/trainingWorkflowData.pkl", 'wb') as file:
    pickle.dump(trainingWorkflowData, file)

