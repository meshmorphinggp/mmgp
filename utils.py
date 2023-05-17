import numpy as np
from scipy import sparse
import networkx
import BasicTools.Containers.ElementNames as ElementNames
import BasicTools.Containers.UnstructuredMeshModificationTools as UMMT
import BasicTools.Containers.UnstructuredMeshInspectionTools as UMIT
import BasicTools.Containers.Filters as Filters

def TruncatedSVDSymLower(matrix, epsilon = None, nbModes = None):
    """
    Computes a truncatd singular value decomposition of a symetric definite
    matrix in scipy.sparse.csr format. Only the lower triangular part needs
    to be defined

    Parameters
    ----------
    matrix : scipy.sparse.csr
        the input matrix
    epsilon : float
        the truncation tolerence, determining the number of keps eigenvalues
    nbModes : int
        the number of keps eigenvalues

    Returns
    -------
    np.ndarray
        kept eigenvalues, of size (numberOfEigenvalues)
    np.ndarray
        kept eigenvectors, of size (numberOfEigenvalues, numberOfSnapshots)
    """

    if epsilon != None and nbModes != None:
        raise("cannot specify both epsilon and nbModes")

    eigenValues, eigenVectors = np.linalg.eigh(matrix, UPLO="L")

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]

    if nbModes == None:
        if epsilon == None:
            nbModes  = matrix.shape[0]
        else:
            nbModes = 0
            bound = (epsilon ** 2) * eigenValues[0]
            for e in eigenValues:
                if e > bound:
                    nbModes += 1
            id_max2 = 0
            bound = (1 - epsilon ** 2) * np.sum(eigenValues)
            temp = 0
            for e in eigenValues:
                temp += e
                if temp < bound:
                    id_max2 += 1

            nbModes = max(nbModes, id_max2)

    if nbModes > matrix.shape[0]:
        print("nbModes taken to max possible value of "+str(matrix.shape[0])+" instead of provided value "+str(nbModes))
        nbModes = matrix.shape[0]

    index = np.where(eigenValues<0)
    if len(eigenValues[index])>0:
        if index[0][0]<nbModes:
            print("removing numerical noise from eigenvalues, nbModes is set to "+str(index[0][0])+" instead of "+str(nbModes))
            nbModes = index[0][0]

    return eigenValues[0:nbModes], eigenVectors[:, 0:nbModes]


def snapshotsPOD_fit_transform(snapshots, correlationOperator, nbModes):
    """Computes and applies the snapshot-POD algorithm

    Parameters
    ----------
    snapshots : np.ndarray
        the input samples for which a reduced dimension approximation is searched
    snapshotCorrelationOperator : scipy.sparse.csr_matrix, optional
        correlation operator between the snapshots
    nbModes : int
        the number of keps eigenvalues

    Returns
    -------
    np.ndarray
        of size (nbModes, nbOfDOFs)
    np.ndarray
        of size (nbModes, nbOfSnapshots)
    """

    numberOfSnapshots = snapshots.shape[0]
    numberOfDofs = snapshots.shape[1]
    correlationMatrix = np.zeros((numberOfSnapshots,numberOfSnapshots))
    matVecProducts = np.zeros((numberOfDofs,numberOfSnapshots))
    for i, snapshot1 in enumerate(snapshots):
        matVecProduct = correlationOperator.dot(snapshot1)
        matVecProducts[:,i] = matVecProduct
        for j, snapshot2 in enumerate(snapshots):
            if j <= i and j < numberOfSnapshots:
                correlationMatrix[i, j] = np.dot(matVecProduct, snapshot2)


    eigenValuesRed, eigenVectorsRed = TruncatedSVDSymLower(correlationMatrix, nbModes = nbModes)

    nbePODModes = eigenValuesRed.shape[0]

    changeOfBasisMatrix = np.zeros((nbePODModes,numberOfSnapshots))
    for j in range(nbePODModes):
        changeOfBasisMatrix[j,:] = eigenVectorsRed[:,j]/np.sqrt(eigenValuesRed[j])

    reducedOrderBasis = np.dot(changeOfBasisMatrix,snapshots)
    generalizedCoordinates = np.dot(reducedOrderBasis, matVecProducts).T
    return reducedOrderBasis, generalizedCoordinates



def RenumberMeshForParametrization(inMesh, inPlace = True, boundaryOrientation = "direct", fixedBoundaryPoints = None, startingPointRankOnBoundary = None):
    """
    Only for linear triangle meshes
    Renumber the node IDs, such that the points on the boundary are placed at the
    end of the numbering. Serves as a preliminary step for mesh parametrization.

    Parameters
    ----------
    inMesh : UnstructuredMesh
        input triangular to be renumbered
    inPlace : bool
        if "True", inMesh is modified
        if "False", inMesh is let unmodified, and a new mesh is produced
    boundaryOrientation : str
        if "direct, the boundary of the parametrisation is constructed in the direct trigonometric order
        if "indirect", the boundary of the parametrisation is constructed in the indirect trigonometric orderc order
    fixedBoundaryPoints : list
        list containing lists of two np.ndarrays. Each 2-member list is used to identify one
        point on the boundary: the first array contains the specified components, and the second the
    startingPointRankOnBoundary : int
        node id (in the complete mesh) of the point on the boundary where the mapping starts

    Returns
    -------
    UnstructuredMesh
        renumbered mesh
    ndarray(1) of ints
        renumbering of the nodes of the returned renumbered mesh, with respect to inMesh
    int
        number of node of the boundary of inMesh

    """
    # assert mesh of linear triangles
    for name, data in inMesh.elements.items():
        name == ElementNames.Triangle_3

    if inPlace == True:
        mesh = inMesh
    else:
        import copy
        mesh = copy.deepcopy(inMesh)

    # Retrieve the elements of the line boundary
    skin = UMMT.ComputeSkin(mesh, md = 2)
    skin.ComputeBoundingBox()


    # Create a path linking nodes of the line boundary, starting with the node with smallest coordinates
    # and going in the direction increasing the value of the second coordinate the least

    bars = skin.elements[ElementNames.Bar_2].connectivity

    nodeGraph0 = ComputeNodeToNodeGraph(skin, dimensionality=1)
    nodeGraph = [list(nodeGraph0[i].keys()) for i in range(nodeGraph0.number_of_nodes())]

    indicesBars = np.sort(np.unique(bars.flatten()))


    if fixedBoundaryPoints == None:

        if startingPointRankOnBoundary == None:
            vec = inMesh.nodes[indicesBars,0]
            indicesNodesXmin = vec == vec[np.argmin(vec)]
            nodesXmin = inMesh.nodes[indicesBars[indicesNodesXmin], :]

            indicesNodesmin = nodesXmin[:,1] == nodesXmin[np.argmin(nodesXmin[:,1]),1]
            nodesmin = nodesXmin[indicesNodesmin, :]


            if inMesh.GetPointsDimensionality() == 3:
                indicesNodesmin = nodesmin[:,2] == nodesmin[np.argmin(nodesmin[:,2]),2]
                nodesmin = nodesmin[indicesNodesmin, :]

            indexInBars = np.where((inMesh.nodes[indicesBars,:] == nodesmin).all(axis=1))[0]
            assert indexInBars.shape == (1,)
            indexInBars = indexInBars[0]
            assert (inMesh.nodes[indicesBars[indexInBars],:] == nodesmin).all()

            pMin = indicesBars[indexInBars]

        else:
            pMin = startingPointRankOnBoundary
        print("starting walking along line boundary at point... =", str(inMesh.nodes[pMin,:]), " of rank:", str(pMin))

    else:
        inds, point = fixedBoundaryPoints[0][0], fixedBoundaryPoints[0][1]
        indexInBars = (np.linalg.norm(np.subtract(inMesh.nodes[indicesBars,:][:,inds], point), axis = 1)).argmin()
        pMin = indicesBars[indexInBars]
        print("starting walking along line boundary at point... =", str(inMesh.nodes[pMin,:]), " of rank:", str(pMin))


    p1 = p1init = pMin
    p2_candidate = [nodeGraph[pMin][0], nodeGraph[pMin][1]]

    if fixedBoundaryPoints == None:
        # choose direction
        p2 = p2_candidate[np.argmin(np.asarray([inMesh.nodes[p2_candidate[0],1], inMesh.nodes[p2_candidate[1],1]]))]

    else:
        # choose direction from second point set on boundary
        inds = fixedBoundaryPoints[1][0]
        delta_fixedBoundaryPoints = fixedBoundaryPoints[1][1] - fixedBoundaryPoints[0][1]
        delta_fixedBoundaryPoints /= np.linalg.norm(delta_fixedBoundaryPoints)

        delta_candidate = np.asarray([inMesh.nodes[p2c,inds] - inMesh.nodes[pMin,inds] for p2c in p2_candidate])
        delta_candidate[0] /= np.linalg.norm(delta_candidate[0])
        delta_candidate[1] /= np.linalg.norm(delta_candidate[1])

        error_delta_candidate = []
        error_delta_candidate.append(np.subtract(delta_candidate[0], delta_fixedBoundaryPoints))
        error_delta_candidate.append(np.subtract(delta_candidate[1], delta_fixedBoundaryPoints))

        p2 = p2_candidate[np.linalg.norm(error_delta_candidate, axis = 1).argmin()]

    print("... walking toward point =", str(inMesh.nodes[p2,:]), " of rank:", str(p2))

    path = [p1, p2]
    while p2 != p1init:
        p2save = p2
        tempArray = np.asarray(nodeGraph[p2])
        p2 = tempArray[tempArray!=p1][0]
        p1 = p2save
        path.append(p2)
    path = path[:-1]

    if boundaryOrientation == "indirect":
        path = path[::-1]

    # Renumber the node, keeping at the end the continuous path along the line boundary
    N = mesh.GetNumberOfNodes()
    nBoundary = len(path)

    initOrder = np.arange(N)
    interiorNumberings = np.delete(initOrder, path)

    renumb = np.hstack((interiorNumberings, path))

    assert len(renumb) == N

    invRenumb = np.argsort(renumb)

    mesh.nodes = mesh.nodes[renumb,:]
    for _, data in mesh.elements.items():
        data.connectivity = invRenumb[data.connectivity]
    mesh.ConvertDataForNativeTreatment()

    return mesh, renumb, nBoundary



def FloaterMeshParametrization(inMesh, nBoundary, outShape = "circle", boundaryOrientation = "direct", curvAbsBoundary = True, fixedInteriorPoints = None, fixedBoundaryPoints = None):
    """
    STILL LARGELY EXPERIMENTAL

    Only for linear triangular meshes

    Computes the Mesh Parametrization algorithm [1] proposed by Floater,
    in the case of target parametrization fitted to the unit 2D circle (R=1) or square (L=1).
    Adapted for ML need: the outShape's boundary is sampled following the curvilinear abscissa along
    the boundary on inMesh (only for outShape = "circle" for the moment)

    Parameters
    ----------
    inMesh : UnstructuredMesh
        Renumbered triangular mesh to parametrize
    nBoundary : int
        number nodes on the line boundary
    outShape : str
        if "circle", the boundary of inMesh is mapped into the unit circle
        if "square", the boundary of inMesh is mapped into the unit square
    boundaryOrientation : str
        if "direct, the boundary of the parametrisation is constructed in the direct trigonometric order
        if "indirect", the boundary of the parametrisation is constructed in the indirect trigonometric order
    curvAbsBoundary : bool
        only if fixedInteriorPoints = None
        if True, the point density on the boundary of outShape is the same as the point density on the boundary of inMesh
        if False, the point density on the boundary is uniform
    fixedInteriorPoints : dict
        with one key, and corresponding value, a list: [ndarray(n), ndarray(n,2)],
        with n the number of interior points to be fixed; the first ndarray is the index of the considered
        interior point, the second ndarray is the corresponding prescribed positions
        if key is "mean", the interior points are displaced by the mean of the prescribed positions
        if key is "value", the interior points are displaced by the value of the prescribed positions
    fixedBoundaryPoints: list
        list of lists: [ndarray(2), ndarray(2)], helping definining a point in inMesh; the first ndarray is the component
        of a point on the boundary, and the second array is the value of corresponding component. Tested for triangular meshes
        in the 3D space.

    Returns
    -------
    UnstructuredMesh
        parametrization of mesh
    dict
        containing 3 keys: "minEdge", "maxEdge" and "weights", with values floats containing the minimal
        and maximal edged length of the parametrized mesh, and the weights (lambda) in the Floater algorithm

    Notes
    -----
        mesh mush be a renumbered UnstructuredMesh of triangles (either in
        a 2D or 3D ambiant space), with a line boundary (no closed surface in 3D).
        outShape = "circle" is more robust in the sense that is inMesh has a 2D square-like,
        for triangles may ended up flat with  outShape = "square"

    References
    ----------
        [1] M. S. Floater. Parametrization and smooth approximation of surface
        triangulations, 1997. URL: https://www.sciencedirect.com/science/article/abs/pii/S0167839696000313
    """
    import copy
    mesh = copy.deepcopy(inMesh)

    N = mesh.GetNumberOfNodes()
    n = N - nBoundary

    u = np.zeros((mesh.nodes.shape[0],2))

    if outShape == "square":
        print("!!! Warning, the implmentation outShape == 'square' is *very* experimental !!!")
        if boundaryOrientation == "indirect":
            raise NotImplementedError("Cannot use 'square' outShape with 'indirect' boundaryOrientation")
        if fixedInteriorPoints != None:
            raise NotImplementedError("Cannot use 'square' outShape with fixedInteriorPoints not None")
        if fixedBoundaryPoints != None:
            raise NotImplementedError("Cannot use 'square' outShape with fixedBoundaryPoints not None")

        # Set the boundary on the parametrization on the unit square
        L = nBoundary//4
        r = nBoundary%4

        u[n:n+L,0] = np.linspace(1/L,1,L)
        u[n:n+L,1] = 0.
        u[n+L:n+2*L,0] = 1.
        u[n+L:n+2*L,1] = np.linspace(1/L,1,L)
        u[n+2*L:n+3*L,0] = np.linspace(1-1/L,0,L)
        u[n+2*L:n+3*L,1] = 1.
        u[n+3*L:n+4*L+r,0] = 0.
        u[n+3*L:n+4*L+r,1] = np.linspace(1-1/(L+r),0,(L+r))

    elif outShape == "circle":
        # Set the boundary on the parametrization on the unit circle

        lengthAlongBoundary = [0]
        cumulativeLength = 0.
        indices = np.arange(n+1, N)
        for i in indices:
            p1 = mesh.nodes[i-1,:]
            p2 = mesh.nodes[i,:]
            cumulativeLength += np.linalg.norm(p2-p1)
            lengthAlongBoundary.append(cumulativeLength)
        lengthAlongBoundary = np.asarray(lengthAlongBoundary)

        if fixedBoundaryPoints != None:
            fixedRanksOnBoundary = [0]
            nFixedPointsOnBoundary = 1
            for fixedBoundaryPoint in fixedBoundaryPoints[1:]:
                inds, point = fixedBoundaryPoint[0], fixedBoundaryPoint[1]
                #indexInBars = np.where((inMesh.nodes[n:,:][:,inds] == point).all(axis=1))[0]
                indexInBars = (np.linalg.norm(np.subtract(inMesh.nodes[n:,:][:,inds], point), axis = 1)).argmin()

                fixedRanksOnBoundary.append(indexInBars)
                nFixedPointsOnBoundary += 1
            fixedRanksOnBoundary.append(-1)

            angles = []
            deltaAngle = 2*np.pi/nFixedPointsOnBoundary
            #print("deltaAngle =", deltaAngle)
            for k in range(nFixedPointsOnBoundary):

                deltaLengthAlongBoundary = lengthAlongBoundary[fixedRanksOnBoundary[k]:fixedRanksOnBoundary[k+1]]-lengthAlongBoundary[fixedRanksOnBoundary[k]]
                deltaUnitLengthAlongBoundary = deltaLengthAlongBoundary/(lengthAlongBoundary[fixedRanksOnBoundary[k+1]]-lengthAlongBoundary[fixedRanksOnBoundary[k]])
                res = (k+deltaUnitLengthAlongBoundary)*deltaAngle
                angles = np.hstack((angles, res))

            angles = np.hstack((angles, 2.*np.pi))

        else:
            if curvAbsBoundary == True:
                angles = (2*np.pi)*(1-1/nBoundary)*lengthAlongBoundary/cumulativeLength
            else:
                angles = np.linspace(2*np.pi/nBoundary, 2*np.pi, nBoundary)

        if boundaryOrientation == "direct":
            for i, a in enumerate(angles):
                u[n+i,0] = np.cos(a)
                u[n+i,1] = np.sin(a)
        else:
            for i, a in enumerate(angles):
                u[n+i,0] = np.cos(a)
                u[n+i,1] = -np.sin(a)

    else:
        raise NotImplementedError("outShape"+str(outShape)+" not implemented")


    # Compute a node graphe for the mesh
    edges = set()
    elFilter = Filters.ElementFilter(mesh, dimensionality = 2, elementTypes = [ElementNames.Triangle_3])
    for name, data ,ids in elFilter:
        for face in ElementNames.faces[name]:
            for idd in ids:
                edge = np.sort(data.connectivity[idd][face[1]])
                edges.add((edge[0], edge[1]))

    G2 = InitializeGraphPointsFromMeshPoints(mesh)
    for edge in edges:
        G2.add_edge(edge[0], edge[1])

    # Compute the weights of each node of the mesh (number of edges linked to each node): the inverse of the degrees
    Ad = networkx.adjacency_matrix(G2)

    weights = np.zeros(N)
    for i in range(N):
        weights[i] = 1./np.sum(Ad[[i],:])

    # Construct the sparse linear system to solve to find the position of the interior points in the parametrization
    A = sparse.eye(n).tolil()
    RHSmat = sparse.lil_matrix((n, N))
    for edge in edges:
        for edg in [(edge[0], edge[1]), (edge[1], edge[0])]:
            if edg[0] < n and edg[1] < n:
                A[edg[0], edg[1]] = -weights[edg[0]]
            elif edg[0] < n:
                RHSmat[edg[0], edg[1]] = weights[edg[0]]

    RHS = RHSmat.dot(u)
    A = A.tocsr()

    # update the position of the interior points
    res = sparse.linalg.spsolve(A, RHS)
    u[:n,:] = res


    if fixedInteriorPoints != None:
        mesh.nodes = u
        mesh.ConvertDataForNativeTreatment()

        displacement = None
        mask = None

        if "mean" in fixedInteriorPoints:

            meanPos = np.mean(u[fixedInteriorPoints["mean"][0],:], axis=0)

            if displacement == None:
                displacement = -np.tile(meanPos,(fixedInteriorPoints["mean"][1].shape[0],1))
            else:
                displacement = np.vstack((displacement, -np.tile(meanPos,(fixedInteriorPoints["mean"][1].shape[0],1))))

            if mask == None:
                mask = fixedInteriorPoints["mean"][0]
            else:
                mask = np.hstack((mask, fixedInteriorPoints["mean"][0]))

        if "value" in fixedInteriorPoints:

            if displacement == None:
                displacement = fixedInteriorPoints["value"][1] - u[fixedInteriorPoints["value"][0],:]
            else:
                displacement = np.vstack((displacement, fixedInteriorPoints["value"][1] - u[fixedInteriorPoints["value"][0],:]))

            if mask == None:
                mask = fixedInteriorPoints["value"][0]
            else:
                mask = np.hstack((mask, fixedInteriorPoints["value"][0]))


        if displacement is not None and mask is not None:
            displacement = np.vstack((displacement, np.zeros((N-n,2))))
            mask = np.hstack((mask, np.arange(n,N)))

            from BasicTools.Containers import UnstructuredMeshModificationTools as UMMT
            new_nodes = UMMT.Morphing(mesh, displacement, mask, radius= 1.)

            mesh.nodes = new_nodes

    else:
        mesh.nodes = u
        mesh.ConvertDataForNativeTreatment()

    infos = {}
    endgeLengths = []
    for edge in edges:
        endgeLengths.append(np.linalg.norm(mesh.nodes[edge[1],:]-mesh.nodes[edge[0],:]))

    infos = {"minEdge":np.min(endgeLengths), "maxEdge":np.max(endgeLengths), "weights":weights}

    return mesh, infos


def ComputeNodeToNodeGraph(inMesh, dimensionality = None, distFunc = None):
    '''Creates a networkx graph from the node connectivity on an UnstructuredMesh through edges

    Parameters
    ----------
    inMesh : UnstructuredMesh
        input mesh
    dimensionality : int
        dimension of the elements considered to initalize the graph
    distFunc : func
        function applied to the lengh of the edges of the mesh, and attached of the
        corresponding edge of the graph of the mesh

    Returns
    -------
    networkx.Graph
        Element to element graph
    '''
    if dimensionality == None:
        dimensionality = inMesh.GetDimensionality()

    if distFunc == None:
        def distFunc(x):
            return x


    elFilter = Filters.ElementFilter(inMesh, dimensionality = dimensionality)
    mesh = UMIT.ExtractElementsByElementFilter(inMesh, elFilter)

    nodeConnectivity, _ = UMIT.ComputeNodeToNodeConnectivity(mesh)

    G = InitializeGraphPointsFromMeshPoints(inMesh)
    edges = []
    for i in range(nodeConnectivity.shape[0]):
        for j in nodeConnectivity[i][nodeConnectivity[i]>i]:
            length = np.linalg.norm(inMesh.nodes[i]-inMesh.nodes[j])
            edges.append((i,j, distFunc(length)))
    G.add_weighted_edges_from(edges)

    return G

def InitializeGraphPointsFromMeshPoints(inMesh):
    '''Initializes a networkx graph with nodes consistant with the number of nodes of an UnstructuredMesh.
    This enables further edge addition compatible with the connectivity of the elements of the UnstructuredMesh.

    Parameters
    ----------
    inMesh : UnstructuredMesh
        input mesh

    Returns
    -------
    networkx.Graph
        initialized graph
    '''
    G = networkx.Graph()
    G.add_nodes_from(np.arange(inMesh.GetNumberOfNodes()))
    return G
