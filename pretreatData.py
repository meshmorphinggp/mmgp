import pickle
import time
import numpy as np
from utils import RenumberMeshForParametrization, FloaterMeshParametrization

from BasicTools.Containers import UnstructuredMeshCreationTools as UMCT
from BasicTools.Containers import UnstructuredMeshFieldOperations as UMFO
from BasicTools.FE import FETools as FT
from BasicTools.FE.Fields import FEField as FF

"""
In this file, the provided dataset is pretreated in the following fashion:
- the meshes are morphed using Floater's algorithm,
- two finite element (FE) interpolation operators are computed:
    - "proj" for the FE interpolation from the current morphed mesh to the common morphed mesh,
    - "invProj" for the FE interpolation from the common morphed mesh to the current morphed mesh,
- The FE interpolation of the spatial coordinate field and output fields of interest is done.

Notice that both FE interpolation operators are precomputed and stored for more time-efficient
simulations in the next stages of the methodology, but examples are provided where the
inference worfkflow is computed from zero.
"""

# Load the provided datasets
file = open("data/datasets.pkl", 'rb')
data = pickle.load(file)
file.close()

inMeshTrain = data["inMeshTrain"]
outFieldsTrain = data["outFieldsTrain"]

inMeshTest = data["inMeshTest"]
outFieldsTest = data["outFieldsTest"]

nOutFields = len(data["outFieldsNames"])

nTrain = len(inMeshTrain)
nTest  = len(inMeshTest)


# Compute the morphing of the median mesh
medianIndex = int(nTrain/2)
medianMesh = UMCT.CreateMeshOfTriangles(inMeshTrain[medianIndex]['points'].astype(np.float32), inMeshTrain[medianIndex]['triangles'])
medianMeshRenumb, renumb, nBoundary = RenumberMeshForParametrization(medianMesh, inPlace = False)
commonProjectionMesh, infos = FloaterMeshParametrization(medianMeshRenumb, nBoundary)
nNodesCommonProjectionMesh = commonProjectionMesh.GetNumberOfNodes()


### TRAINING SET ###

start = time.time()

projOperatorsTrain = []
invProjOperatorsTrain = []
projInCoordFieldsTrain = np.empty((nTrain, 2*nNodesCommonProjectionMesh))
projOutFieldsTrain = np.empty((nTrain, nNodesCommonProjectionMesh, nOutFields))
renumberingTrain = []

for i in range(nTrain):
    print("TRAIN loop, i =", i+1, "over ", nTrain)

    # Compute the morphing of the current mesh
    mesh = UMCT.CreateMeshOfTriangles(inMeshTrain[i]['points'].astype(np.float32), inMeshTrain[i]['triangles'])
    meshRenumb, renumb, nBoundary = RenumberMeshForParametrization(mesh, inPlace = False)
    meshParam, infos = FloaterMeshParametrization(meshRenumb, nBoundary)

    # Compute the two FE projection operators
    renumberingTrain.append(renumb)

    space, numberings, _, _ = FT.PrepareFEComputation(meshParam, numberOfComponents = 1)
    inputFEField = FF.FEField(name="edge",mesh=meshParam, space=space, numbering=numberings[0])
    operator, status = UMFO.GetFieldTransferOp(inputFEField, commonProjectionMesh.nodes, method = "Interp/Clamp", verbose=True)
    projOperatorsTrain.append(operator)

    space, numberings, _, _ = FT.PrepareFEComputation(commonProjectionMesh, numberOfComponents = 1)
    inputFEField = FF.FEField(name="edge",mesh=commonProjectionMesh, space=space, numbering=numberings[0])
    invOperator, status = UMFO.GetFieldTransferOp(inputFEField, meshParam.nodes, method ="Interp/Clamp", verbose=True)
    invProjOperatorsTrain.append(invOperator)

    # Compute the FE interpolation of the spatial coordinate field and output fields of interest
    projInCoordFieldsTrain[i,:] = operator.dot(mesh.nodes[renumb,:]).flatten()
    projOutFieldsTrain[i,:,:] = operator.dot(outFieldsTrain[i][renumb,:])

print("duration for mesh morphing and finite element interpolation =", time.time() - start)


### TESTING SET ###

projOperatorsTest = []
invProjOperatorsTest = []
projInCoordFieldsTest = np.empty((nTest, 2*nNodesCommonProjectionMesh))
projOutFieldsTest = np.empty((nTest, nNodesCommonProjectionMesh, nOutFields))
renumberingTest = []

for i in range(nTest):
    print("TEST loop, i =", i+1, "over ", nTest)

    # Compute the morphing of the current mesh
    mesh = UMCT.CreateMeshOfTriangles(inMeshTest[i]['points'].astype(np.float32), inMeshTest[i]['triangles'])
    meshRenumb, renumb, nBoundary = RenumberMeshForParametrization(mesh, inPlace = False)
    meshParam, infos = FloaterMeshParametrization(meshRenumb, nBoundary)

    # Compute two FE projection operators
    renumberingTest.append(renumb)

    space, numberings, _, _ = FT.PrepareFEComputation(meshParam, numberOfComponents = 1)
    inputFEField = FF.FEField(name="edge",mesh=meshParam, space=space, numbering=numberings[0])
    operator, status = UMFO.GetFieldTransferOp(inputFEField, commonProjectionMesh.nodes, method = "Interp/Clamp", verbose=True)
    projOperatorsTest.append(operator)

    space, numberings, _, _ = FT.PrepareFEComputation(commonProjectionMesh, numberOfComponents = 1)
    inputFEField = FF.FEField(name="edge",mesh=commonProjectionMesh, space=space, numbering=numberings[0])
    invOperator, status = UMFO.GetFieldTransferOp(inputFEField, meshParam.nodes, method ="Interp/Clamp", verbose=True)
    invProjOperatorsTest.append(invOperator)

    # Compute the FE interpolation of the spatial coordinate field and output fields of interest
    projInCoordFieldsTest[i,:] = operator.dot(mesh.nodes[renumb,:]).flatten()
    projOutFieldsTest[i,:,:] = operator.dot(outFieldsTest[i][renumb,:])



# Save the pretreated data

pretreatedData = {}
pretreatedData["projOperatorsTrain"] = projOperatorsTrain
pretreatedData["invProjOperatorsTrain"] = invProjOperatorsTrain
pretreatedData["projInCoordFieldsTrain"] = projInCoordFieldsTrain
pretreatedData["projOutFieldsTrain"] = projOutFieldsTrain
pretreatedData["renumberingTrain"] = renumberingTrain

pretreatedData["projOperatorsTest"] = projOperatorsTest
pretreatedData["invProjOperatorsTest"] = invProjOperatorsTest
pretreatedData["projInCoordFieldsTest"] = projInCoordFieldsTest
pretreatedData["projOutFieldsTest"] = projOutFieldsTest
pretreatedData["renumberingTest"] = renumberingTest

with open("data/pretreatedData.pkl", 'wb') as file:
    pickle.dump(pretreatedData, file)