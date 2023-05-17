import numpy as np
import os, pickle, csv
import matplotlib.pyplot as plt
import matplotlib as mpl

"""
In this file, the code for the bissect graph and the loading characteristics plots
for the 2D solid mechanics case is given (Figures 10 and 11).

Notice that the loading characteristics plots, no precomputation data is used, and the
complete inference workflow is provided and commented.
"""

# Matplotlib plotting options
plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
mpl.style.use("seaborn-v0_8")

fontsize = 32
labelsize = 32
markersize = 24
markeredgewidth = 1

# Clean previous plots
os.system('rm -rf predictScalarsPlots')
os.system('mkdir predictScalarsPlots')

# Load the provided datasets
file = open("data/datasets.pkl", 'rb')
data = pickle.load(file)
file.close()

inMeshTrain = data["inMeshTrain"]
outScalarsTrain = data["outScalarsTrain"]

inMeshTest = data["inMeshTest"]
outScalarsTest = data["outScalarsTest"]
outFieldsTest = data["outFieldsTest"]

outScalarsNames = data["outScalarsNames"]

nOutScalars = len(outScalarsNames)
nTrain = len(data["inScalarsTrain"])
nTest = len(data["inScalarsTest"])


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

correlationOperator1c = trainingWorkflowData["correlationOperator1c"]
correlationOperator2c = trainingWorkflowData["correlationOperator2c"]


# Load the predicted scalars data
file = open("data/predictedScalarsData.pkl", 'rb')
predictedScalarsData = pickle.load(file)
file.close()

predictedOutScalarsTest = predictedScalarsData["predictedOutScalarsTest"]
predictedOutScalarsTestVariance = predictedScalarsData["predictedOutScalarsTestVariance"]
predictedOutScalarsTestQuantile0_025 = predictedScalarsData["predictedOutScalarsTestQuantile0_025"]
predictedOutScalarsTestQuantile0_975 = predictedScalarsData["predictedOutScalarsTestQuantile0_975"]


#### Bissec graph plot ####

labels = [r"$\max_\mathcal{M}(\mathrm{Von~Mises})$",r"$\max_\mathcal{M}(p)$",r"$\max_{\Gamma_{\rm top}}(u_2)$",r"$\max_{\Gamma_{\rm top}}(\sigma_{22})$"]

fig, axs = plt.subplots(2, 2, figsize=(2*6,2*5.5), sharex=False, sharey=False)
axs = axs.ravel()


# Import precomputed GCNN data:
file = open('GCNN/out_salars_target.csv')
csvreader = csv.reader(file)

headerTarget_GCNN = next(csvreader)
rowsTarget_GCNN = np.array([r for r in csvreader], dtype = np.float)
file.close()

file = open('GCNN/out_salars_pred.csv')
csvreader = csv.reader(file)

headerPredict_GCNN = next(csvreader)
rowsPredict_GCNN = np.array([r for r in csvreader], dtype = np.float)
file.close()

assert headerTarget_GCNN == headerPredict_GCNN
assert rowsTarget_GCNN.shape == rowsPredict_GCNN.shape

# Matplotlib instructions
for iax, ax in enumerate(axs):

    y_true_MMGP = outScalarsTest[:,iax]
    y_pred_MMGP = predictedOutScalarsTest[:,iax]

    y_true_GCNN = rowsTarget_GCNN[:,iax]
    y_pred_GCNN = rowsPredict_GCNN[:,iax]

    m = np.min(y_true_MMGP)
    M = np.max(y_true_MMGP)

    ax.plot(np.array([m, M]), np.array([m, M]), color="k")

    ax.plot(y_true_GCNN, y_pred_GCNN, linestyle="", color="r", markerfacecolor="r", markeredgecolor="r",
            markeredgewidth=markeredgewidth, marker=".", markersize=markersize, label=r"$\rm{GCNN}$")

    ax.plot(y_true_MMGP, y_pred_MMGP, linestyle="", color="b",  markerfacecolor="none", markeredgecolor="b",
            markeredgewidth=markeredgewidth, marker=".", markersize=markersize, label=r"$\rm{MMGP}$")

    if iax == 1 or iax == 2:
            ax.set_xscale('log')
            ax.set_yscale('log')

    ax.tick_params(labelsize=labelsize)
    ax.set_title(labels[iax], fontsize=fontsize)

axs[0].set_ylabel(r"$\mathrm{Predictions}$", fontsize=fontsize)
axs[2].set_ylabel(r"$\mathrm{Predictions}$", fontsize=fontsize)
axs[2].set_xlabel(r"$\mathrm{Targets}$", fontsize=fontsize)
axs[3].set_xlabel(r"$\mathrm{Targets}$", fontsize=fontsize)

handles, labels = ax.get_legend_handles_labels()
lgd = axs[0].legend(fontsize=fontsize, columnspacing=0.65, handletextpad=0.1, markerscale=2, loc='lower right')

plt.tight_layout()
fig.savefig("predictScalarsPlots/bissec_plots.png", dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')

print("bissect plot done")


#### Loading characteristics plots ####
###########################################################################################

# Additional imports for the characteristics plots
from BasicTools.Containers import UnstructuredMeshCreationTools as UMCT
from BasicTools.Containers import UnstructuredMeshFieldOperations as UMFO
from BasicTools.FE.Fields import FEField as FF
from BasicTools.FE import FETools as FE
from utils import RenumberMeshForParametrization, FloaterMeshParametrization

nMonteCarloSamples = 10000

# Specify input meshes and pressure
nMeshes = 4
nPointsPerCarac = 11
pressureBoundaries = [-70., -20.]
pressureInputs = pressureBoundaries[0]+(pressureBoundaries[1]-pressureBoundaries[0])/(nPointsPerCarac-1)*np.arange(nPointsPerCarac)

inScalars = np.array([0., 15., 450., 1500., 1500., 7.5e4])

nNodes = []
for i in range(nTest):
    mesh = UMCT.CreateMeshOfTriangles(inMeshTest[i]['points'].astype(np.float32), inMeshTest[i]['triangles'])
    nNodes.append(mesh.GetNumberOfNodes())
nNodes = np.array(nNodes)
rankN = np.argsort(nNodes)
vec = np.arange(len(inMeshTest))
ranks = [vec[rankN[0]], vec[rankN[int(nTest/(nMeshes-1))]], vec[rankN[int(nTest*2/(nMeshes-1))]], vec[rankN[nTest-1]]]

# Compute the morphing of the median mesh
medianIndex = int(nTrain/2)
medianMesh = UMCT.CreateMeshOfTriangles(inMeshTrain[medianIndex]['points'].astype(np.float32), inMeshTrain[medianIndex]['triangles'])
medianMeshRenumb, renumb, nBoundary = RenumberMeshForParametrization(medianMesh, inPlace = False)
commonProjectionMesh, infos = FloaterMeshParametrization(medianMeshRenumb, nBoundary)

# Initialize scalar prediction, variance and quantile arrays
predictedOutScalarsTest = np.empty((nMeshes, nOutScalars, nPointsPerCarac))
predictedOutScalarsTestVariance = np.empty((nMeshes, nOutScalars, nPointsPerCarac))
predictedOutScalarsTestQuantile0_025 = np.empty((nMeshes, nOutScalars, nPointsPerCarac))
predictedOutScalarsTestQuantile0_975 = np.empty((nMeshes, nOutScalars, nPointsPerCarac))


for i, iMesh in enumerate(ranks):

    # Compute the morphing of the current mesh
    mesh = UMCT.CreateMeshOfTriangles(inMeshTest[iMesh]['points'].astype(np.float32), inMeshTest[iMesh]['triangles'])
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
    reducedProjInCoordFields = np.dot(podProjInCoordFields, correlationOperator2c.dot(projInCoordFields.T)).T

    for l in range(nPointsPerCarac):

        inScalars[0] = pressureInputs[l]

        # Rescale the nongeometrical input scalars
        rescaledInScalar = scalerInScalar.transform(inScalars.reshape(1,-1))[0,:]

        # Create the low-dimensional input by concatenating the two previous inputs
        XTest = np.hstack((reducedProjInCoordFields, rescaledInScalar)).reshape(1,-1)

        # Rescale the input
        rescaledXTest = scalerX.transform(XTest)

        predictedProjOutScalarTest = np.empty(nOutScalars)
        predictedProjOutScalarTestSamples = np.empty((nOutScalars, nMonteCarloSamples))
        for j in range(len(outScalarsNames)):
            # Evaluate the scalar regressions
            rescaledYScalarsTest = scalarsRegressors[j].predict(rescaledXTest.reshape(1,-1))[0]
            rescaledYScalarsSamples = scalarsRegressors[j].posterior_samples_f(rescaledXTest.reshape(1,-1), size = nMonteCarloSamples)[0]

            # Inverse rescale the scalar outputs
            predictedOutScalarsTest[i,j,l] = scalerYScalars[j].inverse_transform(rescaledYScalarsTest.reshape(-1, 1))[:,0]
            for k in range(nMonteCarloSamples):
                predictedProjOutScalarTestSamples[j,k] = scalerYScalars[j].inverse_transform(rescaledYScalarsSamples[:,k].reshape(-1, 1))[:,0]

        # Compute variance and quantiles from the Monte Carlo draws
        predictedOutScalarsTestVariance[i,:,l] = np.var(predictedProjOutScalarTestSamples, axis=1)
        predictedOutScalarsTestQuantile0_025[i,:,l] = np.quantile(predictedProjOutScalarTestSamples, 0.025, axis=1)
        predictedOutScalarsTestQuantile0_975[i,:,l] = np.quantile(predictedProjOutScalarTestSamples, 0.975, axis=1)


# Matplotlib instructions
colors = ["tab:red", "tab:blue", "tab:green", "k"]
labels = [r"$\mathrm{Geometry}$ $"+str(i)+"$" for i in range(nMeshes)]
markers = ["o", "s", "^", "P"]
fig1, ax = plt.subplots(1, 1, figsize=(8,7))
fig2, bx = plt.subplots(1, 1, figsize=(8,7))

for i in range(nMeshes):
    ax.errorbar(pressureInputs, predictedOutScalarsTest[i,2,:], yerr=1.96*np.sqrt(predictedOutScalarsTestVariance[i,2,:]), color=colors[i], label=labels[i], marker=markers[i])
    bx.errorbar(pressureInputs, predictedOutScalarsTest[i,0,:], yerr=1.96*np.sqrt(predictedOutScalarsTestVariance[i,0,:]), color=colors[i], label=labels[i], marker=markers[i])

y0 = ax.get_ylim()[0]; dy = 0.02*(ax.get_ylim()[1]-y0)
ax.plot([-50., -40], [y0, y0], color="k", linestyle="-", linewidth = 3)
ax.plot([-50., -50], [y0-dy, y0+dy], color="k", linestyle="-", linewidth = 3)
ax.plot([-40., -40], [y0-dy, y0+dy], color="k", linestyle="-", linewidth = 3)
ax.text(-58, y0+2*dy, r'$\mathrm{training~interval}$', fontsize=fontsize)

y0 = bx.get_ylim()[0]; dy = 0.02*(bx.get_ylim()[1]-y0)
bx.plot([-50., -40], [y0, y0], color="k", linestyle="-", linewidth = 3)
bx.plot([-50., -50], [y0-dy, y0+dy], color="k", linestyle="-", linewidth = 3)
bx.plot([-40., -40], [y0-dy, y0+dy], color="k", linestyle="-", linewidth = 3)
bx.text(-58, y0+2*dy, r'$\mathrm{training~interval}$', fontsize=fontsize)

ax.legend(fontsize=28, bbox_to_anchor=(0.5, 1.3), loc='upper center', ncols=2, columnspacing=0.65, handletextpad=0.1, markerscale=2)
bx.legend(fontsize=28, bbox_to_anchor=(0.5, 1.3), loc='upper center', ncols=2, columnspacing=0.65, handletextpad=0.1, markerscale=2)

xticks = np.arange(np.min(pressureInputs), np.max(pressureInputs)+0.1, 10.0)
ax.set_xticks(xticks)
ax.set_xticklabels([r"$"+str(int(x))+"$" for x in xticks])
bx.set_xticks(xticks)
bx.set_xticklabels([r"$"+str(int(x))+"$" for x in xticks])
ax.tick_params(labelsize=labelsize)
bx.tick_params(labelsize=labelsize)

ax.set_xlabel(r"$\mathrm{Pressure}$", fontsize=fontsize)
bx.set_xlabel(r"$\mathrm{Pressure}$", fontsize=fontsize)
ax.set_ylabel(r"$\max_{\Gamma_{\rm top}}(u_2)$", fontsize=fontsize)
bx.set_ylabel(r"$\max_\mathcal{M}(\mathrm{Von~Mises})$", fontsize=fontsize)


fig1.savefig("predictScalarsPlots/caracs_curves_saved_model_u_2.png", format = "png", bbox_inches="tight")
fig2.savefig("predictScalarsPlots/caracs_curves_saved_model_VonMises.png", format = "png", bbox_inches="tight")

print("loading characteristics plot done")
