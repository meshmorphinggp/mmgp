import numpy as np
import os, pickle
from BasicTools.IO import GeofReader as GR
from BasicTools.IO import UtReader as UR
from env import *


folder = "/gpfs_new/cold-data/InputData/public_datasets/2D_Meca_EVP/raw"

outFieldsNames = ["U1", "U2", "evrcum", "sig11", "sig22", "sig12"]
inScalarsNames = ["P", "R0", "K", "C", "D", "young"]
outScalarsNames = ["max_von_mises", "max_evrcum", "max_U2_top", "max_sig22_top"]



inMeshTrain = []
inScalarsTrain = np.load("../genEvpZset//DOE_TRAIN_MaxProj_7_500.npy")
nTrain = inScalarsTrain.shape[0]
outScalarsTrain = np.empty((nTrain, len(outScalarsNames)))
outFieldsTrain = []


#nTrain=1
for i in range(nTrain):
    print("TRAIN loop, i =", i+1, "over ", nTrain)

    # inMeshTrain
    inMeshTrain.append({})
    mesh = GR.ReadGeof(folder+"/train/"+str(i)+"/geometry_"+str(i)+"_out.geof")
    mesh.elements['tri3'].connectivity = mesh.elements['tri3'].connectivity[:,[0,2,1]]
    mesh.ConvertDataForNativeTreatment()

    inMeshTrain[i]['points'] = mesh.nodes
    inMeshTrain[i]['triangles'] = mesh.elements['tri3'].connectivity

    nNodes = mesh.GetNumberOfNodes()

    # outFieldsTrain
    outFieldsTrain.append(np.empty((nNodes, len(outFieldsNames))))
    for j, name in enumerate(outFieldsNames):
        outFieldsTrain[i][:,j] = UR.ReadFieldFromUt(fileName=folder+"/train/"+str(i)+"/compute_"+str(i)+".ut", fieldname=name, timeIndex=1, atIntegrationPoints=False)


    # outScalarsTrain
    bar_ids = mesh.elements['bar2'].GetTag("Top").GetIds()
    top_ids = np.unique(np.ravel(mesh.elements['bar2'].connectivity[bar_ids,:]))


    s11 = outFieldsTrain[i][:,3]
    s22 = outFieldsTrain[i][:,4]
    s12 = outFieldsTrain[i][:,5]
    outScalarsTrain[i,0] =  np.max(np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2))
    outScalarsTrain[i,1]  = np.max(outFieldsTrain[i][:,2])
    outScalarsTrain[i,2]  = np.max(outFieldsTrain[i][top_ids,1])
    outScalarsTrain[i,3] = np.max(outFieldsTrain[i][top_ids,4])

inMeshTest = []
inScalarsTest = np.load("../genEvpZset/DOE_TEST_MaxMin_7_200.npy")
nTest = inScalarsTest.shape[0]
outScalarsTest = np.empty((nTest, len(outScalarsNames)))
outFieldsTest = []


#nTest=1
for i in range(nTest):
    print("TEST loop, i =", i+1, "over ", nTest)

    # inMeshTest
    inMeshTest.append({})
    mesh = GR.ReadGeof(folder+"/test/"+str(i)+"/geometry_"+str(i)+"_out.geof")
    mesh.elements['tri3'].connectivity = mesh.elements['tri3'].connectivity[:,[0,2,1]]
    mesh.ConvertDataForNativeTreatment()

    inMeshTest[i]['points'] = mesh.nodes
    inMeshTest[i]['triangles'] = mesh.elements['tri3'].connectivity

    nNodes = mesh.GetNumberOfNodes()

    # outFieldsTest
    outFieldsTest.append(np.empty((nNodes, len(outFieldsNames))))
    for j, name in enumerate(outFieldsNames):
        outFieldsTest[i][:,j] = UR.ReadFieldFromUt(fileName=folder+"/test/"+str(i)+"/compute_"+str(i)+".ut", fieldname=name, timeIndex=1, atIntegrationPoints=False)


    # outScalarsTest
    bar_ids = mesh.elements['bar2'].GetTag("Top").GetIds()
    top_ids = np.unique(np.ravel(mesh.elements['bar2'].connectivity[bar_ids,:]))


    s11 = outFieldsTest[i][:,3]
    s22 = outFieldsTest[i][:,4]
    s12 = outFieldsTest[i][:,5]
    outScalarsTest[i,0] =  np.max(np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2))
    outScalarsTest[i,1]  = np.max(outFieldsTest[i][:,2])
    outScalarsTest[i,2]  = np.max(outFieldsTest[i][top_ids,1])
    outScalarsTest[i,3] = np.max(outFieldsTest[i][top_ids,4])

print("===")

inMeshTestNewForm = []
inScalarsTestNewForm = np.empty((2, len(inScalarsNames)))
outScalarsTestNewForm = np.empty((2, len(outScalarsNames)))
outFieldsTestNewForm = []


bounds = {}
bounds['P'] = (-50., -40.)
bounds['R0'] = (10., 20.)
bounds['K'] = (300., 600.)
bounds['D'] = (1000., 2000.)
bounds['C'] = (1000., 2000.)
bounds['young'] = (5.e4, 1.e5)



# inScalarsTest
for i in range(2):
    for j, name in enumerate(inScalarsNames):
        inScalarsTestNewForm[i,j] = np.round((bounds[name][0]  + bounds[name][1]) /2, 3)


# inMeshTest Ellipsoid
mesh = GR.ReadGeof("/gpfs_new/cold-data/InputData/public_datasets/2D_Meca_EVP/raw/newForms/ellipsoid/geometry_out.geof")
mesh.elements['tri3'].connectivity = mesh.elements['tri3'].connectivity[:,[0,2,1]]
mesh.ConvertDataForNativeTreatment()

inMeshTestNewForm.append({})
inMeshTestNewForm[0]['points'] = mesh.nodes
inMeshTestNewForm[0]['triangles'] = mesh.elements['tri3'].connectivity

nNodes = mesh.GetNumberOfNodes()

# outFieldsTest Ellipsoid
outFieldsTestNewForm.append(np.empty((nNodes, len(outFieldsNames))))
for j, name in enumerate(outFieldsNames):
    outFieldsTestNewForm[0][:,j] = UR.ReadFieldFromUt(fileName="/gpfs_new/cold-data/InputData/public_datasets/2D_Meca_EVP/raw/newForms/ellipsoid/compute.ut", fieldname=name, timeIndex=1, atIntegrationPoints=False)

# outScalarsTest Ellipsoid
bar_ids = mesh.elements['bar2'].GetTag("Top").GetIds()
top_ids = np.unique(np.ravel(mesh.elements['bar2'].connectivity[bar_ids,:]))

s11 = outFieldsTestNewForm[0][:,3]
s22 = outFieldsTestNewForm[0][:,4]
s12 = outFieldsTestNewForm[0][:,5]
outScalarsTestNewForm[0,0] =  np.max(np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2))
outScalarsTestNewForm[0,1]  = np.max(outFieldsTestNewForm[0][:,2])
outScalarsTestNewForm[0,2]  = np.max(outFieldsTestNewForm[0][top_ids,1])
outScalarsTestNewForm[0,3] = np.max(outFieldsTestNewForm[0][top_ids,4])


# inMeshTest Wedge
mesh = GR.ReadGeof("/gpfs_new/cold-data/InputData/public_datasets/2D_Meca_EVP/raw/newForms/wedge/geometry_out.geof")
mesh.elements['tri3'].connectivity = mesh.elements['tri3'].connectivity[:,[0,2,1]]
mesh.ConvertDataForNativeTreatment()

inMeshTestNewForm.append({})
inMeshTestNewForm[1]['points'] = mesh.nodes
inMeshTestNewForm[1]['triangles'] = mesh.elements['tri3'].connectivity

nNodes = mesh.GetNumberOfNodes()

# outFieldsTest Wedge
outFieldsTestNewForm.append(np.empty((nNodes, len(outFieldsNames))))
for j, name in enumerate(outFieldsNames):
    outFieldsTestNewForm[1][:,j] = UR.ReadFieldFromUt(fileName="/gpfs_new/cold-data/InputData/public_datasets/2D_Meca_EVP/raw/newForms/wedge/compute.ut", fieldname=name, timeIndex=1, atIntegrationPoints=False)

# outScalarsTest Wedge
bar_ids = mesh.elements['bar2'].GetTag("Top").GetIds()
top_ids = np.unique(np.ravel(mesh.elements['bar2'].connectivity[bar_ids,:]))


s11 = outFieldsTestNewForm[1][:,3]
s22 = outFieldsTestNewForm[1][:,4]
s12 = outFieldsTestNewForm[1][:,5]
outScalarsTestNewForm[1,0] =  np.max(np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2))
outScalarsTestNewForm[1,1]  = np.max(outFieldsTestNewForm[0][:,2])
outScalarsTestNewForm[1,2]  = np.max(outFieldsTestNewForm[0][top_ids,1])
outScalarsTestNewForm[1,3] = np.max(outFieldsTestNewForm[0][top_ids,4])


# Save Data
data = {}

data["outFieldsNames"] = outFieldsNames
data["inScalarsNames"] = inScalarsNames
data["outScalarsNames"] = outScalarsNames

for i in range(nTrain):
    inMeshTrain[i]['points'] = inMeshTrain[i]['points'].astype(np.float16)
data["inMeshTrain"] = inMeshTrain
data["inScalarsTrain"] = inScalarsTrain
data["outScalarsTrain"] = outScalarsTrain
data["outFieldsTrain"] = [f.astype(np.float16) for f in outFieldsTrain]

for i in range(nTest):
    inMeshTest[i]['points'] = inMeshTest[i]['points'].astype(np.float16)
data["inMeshTest"] = inMeshTest
data["inScalarsTest"] = inScalarsTest
data["outScalarsTest"] = outScalarsTest
data["outFieldsTest"] = [f.astype(np.float16) for f in outFieldsTest]

for i in range(2):
    inMeshTestNewForm[i]['points'] = inMeshTestNewForm[i]['points'].astype(np.float16)
data["inMeshTestNewForm"] = inMeshTestNewForm
data["inScalarsTestNewForm"] = inScalarsTestNewForm
data["outScalarsTestNewForm"] = outScalarsTestNewForm
data["outFieldsTestNewForm"] = [f.astype(np.float16) for f in outFieldsTestNewForm]

with open("data/datasets.pkl", 'wb') as file:
    pickle.dump(data, file)

os.system("tar -c -J -f data/datasets.tar.xz data/datasets.pkl")