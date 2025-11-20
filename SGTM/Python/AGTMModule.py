import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
from scipy.interpolate import CubicSpline
import pickle as pkl
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from scipy.io import loadmat
from pathlib import Path
from copy import deepcopy

def Standardized_AGTM_InitTrain(FG, F_NoisyData, IntDim, r, epsilon, mem):
    '''
    Standardized_AGTM_InitTrain sets up AGTM and performs training calling
    AGTM_EM after standardization of datasets and graphs.
    '''

    F_NoisyData = F_NoisyData.astype(float)
    k = 0
    Nets = []
    LogL = []
    GMDist = []
    NoisyMan = []
    for i in range(len(FG)):
        
        print("net n.%s" % str(i))
        print(FG[i])

        Adj = nx.adjacency_matrix(FG[i]).toarray()
        Nodes = AllNodeNames(FG[i])
        NN = rangesearch(F_NoisyData, Nodes, r)
        T = F_NoisyData[np.unique(np.hstack(NN.T)), :]

        #G = nx.from_numpy_matrix(Adj)
        G = nx.DiGraph(Adj)

        r_AGTM = np.mean(np.min(cdist(Nodes, Nodes) + 1e10 * np.identity(len(Nodes)), axis=0))

        Netk = gtminit(G, Nodes, IntDim, epsilon, T, r)

        # Intialization ends here
        Netk, logLk = agtm_em(Netk, 1, T, mem)

# self.Type = Type
#         self.LatentD = LatentD
#         self.graph = graph
#         self.gmmnet = gmmnet
#         self.rbfnet = rbfnet
            
        Nets.append(Netk)
        LogL.append(logLk)

        V_ = Netk.gmmnet.V_
        Sigma = Netk.gmmnet.Sigma
        invSigma = multinv(Sigma)
        priors = np.ravel(Netk.gmmnet.priors)

        GMDistk = GaussianMixture(n_components=len(V_), max_iter=1, weights_init=priors, means_init=V_,
                                  precisions_init=invSigma.T)

        GMDist.append(GMDistk)
        NoisyMan.append(T)

        k = k + 1

    return Nets, LogL, GMDist, NoisyMan


def AllNodeNames(G):
    '''
    Stacks all Node names into an nd-array
    '''

    Nodes = list(G.nodes(data=True))

    Names = np.zeros((len(Nodes), 3))
    for i, n in enumerate(G.nodes):
        Names[i, :] = np.frombuffer(G.nodes[n]['Name'])

    return Names


def gtminit(Graph, Nodes, LatentD, c_width, Data, r):
    '''
    GTMINIT initializes the AGTM net using the graph, latent dimension, size
    of RBFs kernels, noisy data set and neighbourhood search radius inputs.

    Description:
    This functions firstly sets up the AGTM net. It then procedes to
    compute a the weighted covariance matrix of a small neighbourhood
    centered at each node of the graph.
    '''

    Net = agtm(Graph, Nodes, LatentD, c_width)

    V_ = Net.gmmnet.V_
    m, n = np.shape(V_)

    KCov_width = 2 * r

    N = rangesearch(Data, V_, KCov_width)

    Sigma1 = np.ones((n, n, m))
    Sigma = np.ones((n, n, m))

    i = 0


    for i in range(len(N)):
        Sigma1[:, :, i] = KCov(V_[i, :], Data[N[i], :], KCov_width)
        Sigma1[:, :, i] = 0.5 * (Sigma1[:, :, i] + Sigma1[:, :, i].T) + 1e-5 * np.identity(n)
        Sigma[:, :, i] = np.cov(Data[N[i], :].T)
        Sigma[:, :, i] = 0.5 * (Sigma[:, :, i] + Sigma[:, :, i].T)

        if np.isnan(np.sum(Sigma1[:, :, i])) or np.linalg.cond(Sigma1[:, :, i]) > 1e3:
            Sigma1[:, :, i] = np.identity(3) * 1e-2

    
    Net.gmmnet.T = Data
    Net.gmmnet.Sigma1 = Sigma1
    Net.gmmnet.Sigma = Sigma
    Net.gmmnet.KCov_width = KCov_width
    Net.gmmnet.priors = np.ones((len(V_), 1)) * (1 / len(V_))
    Net.gmmnet.alpha = 1

    return Net


def multinv(M):
    '''
    Equivalent of MATLAB's multinv function:
    Author: Xiaodong Qi <i2000s@hotmail.com>
    History: original 26-Apr-2011
    Inspired by Bruno Luong's MultiSolver code to solve a linear equ system.

    Copyright (c) 2011, Xiaodong Qi
    All rights reserved.

    Inverses each 2D slice of an array (M) with arbitrary dimensions support.
    '''
    sn = np.shape(M)
    m = sn[0]
    n = sn[1]

    if m != n:
        raise NameError("multinv: The first two dimensions of M must be m x m slices.")

    p = np.prod(sn[2:])
    M = np.reshape(M, (m, n, p))

    # Build sparse matrix and solve
    I = np.reshape(np.array(range(m * p)), (m, 1, p), order='F')  # column-major search
    I = np.tile(I, (1, n, 1))  # m x n x p
    I = I.flatten(order='F')

    J = np.reshape(np.array(range(n * p)), (1, n, p), order='F')  # column-major search
    J = np.tile(J, (m, 1, 1))  # m x n x p
    J = J.flatten(order='F')

    M = coo_matrix((M.flatten(order='F'), (I, J))).toarray()

    RHS = np.tile(np.identity(m), (p, 1))
    X = np.linalg.solve(M, RHS)
    X = np.reshape(X, (n, p, m), order='F')
    X = np.transpose(X, [0, 2, 1])
    X = np.reshape(X, (n, m, sn[2:][0]))

    return X


def agtm_em(Net, NIter, T, mem):
    '''
    AGTM_EM performs E-M algorithm on the struct array net using the data
    in T as training data set.

    Description:
    This function computes the parameters updates for AGTM at each
    iteration. Parameters W and alpha are updated by computing the
    responsibilities of net.gmmnet w.r.t. to data set T. This is performed
    taking advantage of the GMDISTRIBUTION function in MATLAB.
    At each iteration the new centers and covariance matrices are
    substitued to the provious ones. The outputs are the updated AGTM and
    the logL function. logL can be used to inspect convergence.
    '''

    priors = Net.gmmnet.priors
    priors = priors.reshape(len(priors))
    V_ = Net.gmmnet.V_

    Sigma = Net.gmmnet.Sigma1
    N, D = np.shape(T)
    temp = Sigma
    SigmaNew = Sigma
    Norm = 1. / D

    Phi = Net.rbfnet.Phi

    DimM, K = np.shape(Phi)

    invSigma = multinv(Sigma)

    print(priors)
    # GM = GaussianMixture(n_components=len(V_), max_iter=1, weights_init=priors, means_init=V_, precisions_init=invSigma.T).fit(T)

    # R = GM.predict_proba(T)

    logL, Norm_logL, R = Compute_logL(V_, Sigma, T)

    LogL = np.zeros(NIter)
    LogL[0] = logL

    W = Net.gmmnet.W
    alpha = np.tile(Net.gmmnet.alpha, (1, np.shape(R)[1]))

    T2 = np.repeat(T.T[:, :, np.newaxis], K, axis=2)
    Phi2 = Phi.reshape((1, DimM, K), order='F')

    for i in range(NIter):

        # E-step
        # GM = GaussianMixture(n_components=len(V_), max_iter=1, weights_init=priors, means_init=V_, precisions_init=invSigma.T).fit(T)

        # R = GM.predict_proba(T)
        logL, Norm_logL, R = Compute_logL(V_, Sigma, T)
        prodT = np.matmul(T.T, R)

        # M-step
        print(i)
        G = np.sum(R, axis=0)
        invSigma = invSigma * (1. / alpha)
        Kron2 = invSigma * G
        KronS = np.zeros((D * DimM, D * DimM))

        # W update (three different ways for computing the product in the
        # equation. They should be equivalent). Suggested mem==0, that is quite
        # memory intensive but faster (especially on low-dimensional problems).

        if mem == 0:
            for j in range(DimM):
                Psi_i = np.reshape(Phi[j, :] * Phi, (DimM, 1, 1, K), order='F')
                temp1 = np.reshape(Kron2, (1, D, D, K), order='F')
                temp2 = np.einsum('ijkl,abcl->ibcl', Psi_i, temp1)
                temp3 = np.sum(temp2, axis=3)
                temp4 = np.transpose(temp3, [1, 2, 0])
                temp5 = np.reshape(temp4, (D, DimM * D), order='F')
                KronS[(D * j):D * (j + 1), :] = temp5
        # print(KronS)

        elif mem == 1:
            Psi_i = np.reshape(Phi, (DimM, 1, K), order='F')
            temp1 = np.reshape(Kron2, (D * D, K), order='F')
            temp2 = np.einsum('ijk,ak->iak', Psi_i, temp1)
            Const_Kron = np.transpose(temp2, [2, 1, 0])

            temp3 = np.einsum('ij,jkl->ikl', Phi, Const_Kron)
            temp4 = np.reshape(temp3, (DimM, D, D, DimM), order='F')
            temp5 = np.transpose(temp4, [1, 0, 2, 3])
            KronS = np.reshape(temp5, (DimM * D, DimM * D), order='F')
        # print(KronS)

        else:
            for k in range(K):
                temp = Phi[:, k].reshape((len(Phi[:, k]), 1))
                Psi = np.dot(temp, temp.T)
                KronS = KronS + np.kron(Psi.T, Kron2[:, :, k])

        # print(KronS)

        # Compute Right hand side of the W update equation.
        R2 = np.reshape(R, (N, 1, K), order='F')
        prod1 = np.einsum('mnr,ndr->mdr', invSigma, T2)
        prod2 = np.einsum('mnr,ndr->mdr', prod1, R2)
        prod3 = np.einsum('mnr,ndr->mdr', prod2, Phi2)

        rhs = np.sum(prod3, axis=2)

        # Invert the Kron matrix by svd.
        U, S, Vh = np.linalg.svd(KronS)
        V = Vh.T

        s = np.diag(S)
        tolerance = 1e-4
        p = np.sum(s > tolerance)

        Up = U[:, 0:p]
        Vp = V[:, 0:p]

        Inv = np.diag(1. / S[0:p])
        AInv = np.matmul(np.matmul(Vp, Inv), Up.T)

        x = np.matmul(AInv, rhs.flatten(order='F'))
        W = np.reshape(x, (D, DimM), order='F').T

        # Recompute the new centers of gaussian mixture.
        V_ = np.matmul(Phi.T, W)

        print(V_)
        # alpha update
        d = mahal_d(T, V_, SigmaNew)

        M = R * d

        alpha = Norm * np.sum(M, axis=0) / G
        alpha = alpha.reshape((1, len(alpha)))

        Max = np.maximum(alpha.T, 1e-1 * np.ones((np.shape(alpha)[1], 1)))
        alpha = np.minimum(Max, np.ones((np.shape(alpha)[1], 1)) * 2).T

        SigmaNew = Sigma * alpha

        for j in range(np.shape(SigmaNew)[2]):
            EigDec, EigVec = np.linalg.eig(SigmaNew[:, :, j])
            if check_symmetric(SigmaNew[:, :, j]) == False or np.any(EigDec < 0):
                SigmaNew[:, :, j] = 0.5 * SigmaNew[:, :, j] + SigmaNew[:, :, j].T

        # Update the gaussian mixture model
        InvSigmaNew = multinv(SigmaNew)
    # GMNew = GaussianMixture(n_components=len(V_), max_iter=1, weights_init=priors, means_init=V_, precisions_init=InvSigmaNew.T).fit(T)

    # logLi = GMNew.score(T)
    # LogL[i] = logLi

    Net.gmmnet.alpha = alpha
    Net.gmmnet.V_ = V_
    Net.gmmnet.W = W
    Net.gmmnet.Sigma = SigmaNew

    # Net.gmmnet.GMdist = GMNew

    return Net, LogL


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    '''
    Checks if given matrix 'a' is symmetric
    '''
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def rangesearch(X, Y, Radius, return_distances=False):
    '''
    - Short-hand for applying MATLAB's rangesearch function
    for finding indices and distances to nearest neighbors (NN).
    - Default for returning distances is False.
    '''

    tree = KDTree(X)
    NN = tree.query_radius(Y, Radius, return_distances)

    return NN


def KCov(mu, NData, sigma2):
    '''
    KCOV  Computes the covariance matrix for each point found through
    crawling considering it as the centres of a kernel.

    Description:

    Input parameters:
    - mu: point estimated through crawling.
    - NData: Training data for GTM.
    - sigma2: sigma squared associated to the kernel (multiple of the
      crawling length).

    Output parameters:
    - C: covariance matrix relative to point mu.
    '''

    m, n = np.shape(NData)
    mu = np.tile(mu, (m, 1))
    diff = NData - mu

    NDiff = np.zeros((len(diff), 1))

    for i in range(len(diff)):
        NDiff[i] = np.linalg.norm(diff[i, :])

    k = np.exp(-(NDiff ** 2) / (2 * sigma2 ** 2))
    KSum = np.sum(k)

    prod = np.zeros((m, n, n))

    for i in range(m):
        a = diff[i, :].reshape((len(diff[i, :]), 1))
        b = diff[i, :].reshape((1, len(diff[i, :])))

        prod[i, :, :] = k[i] * np.dot(a, b)

    C = np.sum(prod, axis=0).reshape((n, n))
    C = C / KSum

    return C


def Compute_logL(means, Cov, ED):
    normal = (2 * np.pi) ** (np.shape(ED)[1] / 2.)

    # Compute Mahalanobis distance between any point in ED and any center of GM

    Mahal_d = mahal_d(ED, means, Cov)

    a = np.exp(-0.5 * Mahal_d)

    DetC = np.zeros((np.shape(Cov)[2], 1))

    for k in range(np.shape(Cov)[2]):
        DetC[k] = np.linalg.det(Cov[:, :, k])

    # Divide by determinant of every component's covariance matrix
    act = a / np.tile(normal * np.sqrt(DetC.T), (len(a), 1))

    # Multiply the sum of all activations by the prior (1/NumComp)
    l1 = np.log(np.sum(act, axis=1) / np.shape(a)[1])

    logL = np.sum(l1)
    Norm_logL = logL / len(ED)

    # Calculate posterior
    post = act * np.shape(a)[1]
    s = np.sum(post, axis=1)
    s = s + (s == 0)
    s = s.reshape((len(s), 1))

    post = post / np.dot(s, np.ones((1, len(means))))

    return logL, Norm_logL, post


def mahal_d(obs, mean, cov_M):
    '''
    MAHAL_D computes the mahalanobis distance between every center in MEAN
    and the noisy data points in obs, using the covariance matrices in cov_M.

    Description:
    This function firstly decomposes the covariance matrix using Choleski
    decomposition to avoid singularities and speed up the inversion task.
    For every mean it computes the mahalanobis distance of the observed
    points w.r.t. the mean.
    '''
    d = np.zeros((len(obs), len(mean)))

    for i in range(len(mean)):

        diff = obs - np.ones((len(obs), 1)) * mean[i, :]

        try:

            c = np.linalg.cholesky(cov_M[:, :, i]).T
            temp = np.matmul(diff, np.linalg.inv(c))
            d[:, i] = np.sum(temp * temp, axis=1)



        except:
            print("gtmem: Warning -- Cov matrix singular, using pinv.")
            S_1 = np.linalg.pinv(cov_M[:, :, i])
            d[:, i] = np.sum(np.matmul(diff, S_1) * diff, axis=1)

    return d


def multinv(M):
    '''
    Equivalent of MATLAB's multinv function:
    Author: Xiaodong Qi <i2000s@hotmail.com>
    History: original 26-Apr-2011
    Inspired by Bruno Luong's MultiSolver code to solve a linear equ system.

    Copyright (c) 2011, Xiaodong Qi
    All rights reserved.

    Inverses each 2D slice of an array (M) with arbitrary dimensions support.
    '''
    sn = np.shape(M)
    m = sn[0]
    n = sn[1]

    if m != n:
        raise NameError("multinv: The first two dimensions of M must be m x m slices.")

    p = np.prod(sn[2:])
    M = np.reshape(M, (m, n, p))

    # Build sparse matrix and solve
    I = np.reshape(np.array(range(m * p)), (m, 1, p), order='F')  # column-major search
    I = np.tile(I, (1, n, 1))  # m x n x p
    I = I.flatten(order='F')

    J = np.reshape(np.array(range(n * p)), (1, n, p), order='F')  # column-major search
    J = np.tile(J, (m, 1, 1))  # m x n x p
    J = J.flatten(order='F')

    M = coo_matrix((M.flatten(order='F'), (I, J))).toarray()

    RHS = np.tile(np.identity(m), (p, 1))
    X = np.linalg.solve(M, RHS)
    X = np.reshape(X, (n, p, m), order='F')
    X = np.transpose(X, [0, 2, 1])
    X = np.reshape(X, (n, m, sn[2:][0]))

    return X


class net:

    def __init__(self, Type, LatentD, graph):
        self.Type = Type
        self.LatentD = LatentD
        self.graph = graph
        self.gmmnet = gmmnet([],[],[],[],[],[],[],[],[],[],[])
        self.rbfnet = rbfnet([],[],[],[],[],[],[],[])
        

def agtm(Graph, Nodes, LatentD, c_width):
    '''
    AGTM builds the struct array net, containing all AGTM fields.

    Description:
    This functions sets the AGTM net struct array.
    The net array contains the RBFs net and GMMNET net plus information on
    the global setup of AGTM:
    - Initialization Graph
    - Latent dimensionality
    - Type of activation function
    '''

    Net_aux = net([],[],[])

    # net.Type = "AGTM"
    # net.LatentD = LatentD
    # net.graph = Graph

    Net_aux.Type = "AGTM"
    Net_aux.LatentD = LatentD
    Net_aux.graph = Graph

    # setup rbfnet

    actfcn = "Gaussian"
    Adj = nx.adjacency_matrix(Graph).toarray()
    #LGraph = nx.from_numpy_matrix(Adj)
    LGraph = nx.DiGraph(Adj)

    #net.rbfnet = rbf(LGraph, LatentD, c_width, actfcn)
    Net_aux.rbfnet = rbf(LGraph, LatentD, c_width, actfcn)

    #net.gmmnet = gmm(Graph, Nodes, LatentD)
    #Net_aux.gmmnet = gmm(Graph, Nodes, LatentD)

    Net_aux.gmmnet.Type = "gmm"
    Net_aux.gmmnet.LatentD = LatentD
    Net_aux.gmmnet.V_ = Nodes

    #Net = rbffwd(net)
    Net = rbffwd(Net_aux)

    return Net


def rbffwd(Net):
    '''
    RBFFWD propagates the abstract graph nodes onto the embedding via the
    RBF network.

    Description:
       This function propagates the abstract graph nodes onto the emebedding
       using the RBF network. The new embedded nodes are stired in the field
       GMMNET.
    '''

    gmm_aux = Net.gmmnet

    rbfnet_aux = Net.rbfnet

    V_ = gmm_aux.V_
    Phi = rbfnet_aux.Phi

    Lambda = 1e-5

    H = np.dot(Phi, Phi.T)

    W = np.linalg.solve(H + Lambda * np.ones(len(H)), Phi)
    W = np.dot(W, V_)

    Net.gmmnet.W = W
    Net.gmmnet.V_ = np.dot(W.T, Phi).T

    return Net

class gmmnet:

    def __init__(self, Type, LatentD, V_, W, T, Sigma1, Sigma, KCov_width, priors, alpha, GMdist):
        self.Type = Type
        self.LatentD = LatentD
        self.V_ = V_
        self.W = W
        self.T = T
        self.Sigma1 = Sigma1
        self.Sigma = Sigma
        self.KCov_width = KCov_width
        self.priors = priors
        self.alpha = alpha
        self.GMdist = GMdist

def gmm(G, Nodes, LatentD):
    '''
    GMM sets the field GMMNET in AGTM.

    Description:
       This function is only used once during initialization.
       It sets the latent dimension field and the initial centers (the nodes
       of graph G).
    '''

    gmmnet_aux = gmmnet([],[],[],[],[],[],[],[],[],[],[])

    #gmmnet.Type = "gmm"
    #gmmnet.LatentD = LatentD
    #gmmnet.V_ = Nodes

    gmmnet_aux.Type = "gmm"
    gmmnet_aux.LatentD = LatentD
    gmmnet_aux.V_ = Nodes

    #return gmmnet
    return gmmnet_aux



class rbfnet:

    def __init__(self, Type, actfcn, LatentD, graph, c, c_width, V, Phi):
        self.Type = Type
        self.actfcn = actfcn
        self.LatentD = LatentD
        self.graph = graph
        self.c = c
        self.c_width = c_width
        self.V = V
        self.Phi = Phi

def rbf(G, LatentD, c_width, actfcn):
    '''
    RBF sets the RBF field in AGTM.

    Description:
    This functions sets the Radial Basis Function net as a field of AGTM.
    It first callas GRAPH_EMESH to identify the centers of the RBFs, then
    it computes the geodesic distances of all nodes on abstract graph G
    w.r.t. these centers and computes the activation functions.
    '''

    rbfnet_aux = rbfnet([],[],[],[],[],[],[],[])

    #rbfnet.Type = "RBFnet"
    #rbfnet.actfcn = actfcn
    #rbfnet.LatentD = LatentD
    #rbfnet.graph = G

    rbfnet_aux.Type = "RBFnet"
    rbfnet_aux.actfcn = actfcn
    rbfnet_aux.LatentD = LatentD
    rbfnet_aux.graph = G

    # Build the epsilon-mesh on the graph
    RBF_c, MInd = Graph_EMesh(G, c_width, LatentD)
    #rbfnet.c = RBF_c
    rbfnet_aux.c = RBF_c

    # Recover the size of the basis function centers (geodesic distances
    # between nodes of the graph)
    cdist = distances(G, RBF_c, RBF_c)

    cdist = cdist + 1e20 * np.identity(len(RBF_c))
    c_width = np.min(cdist)

    #rbfnet.c_width = c_width
    rbfnet_aux.c_width = c_width

    V = np.array(range(len(G.nodes)))
    #rbfnet.V = V.reshape((len(V), 1))
    rbfnet_aux.V = V.reshape((len(V), 1))

    Dist = distances(G, RBF_c, V.reshape((len(V), 1)))

    factor = 3

    Dist2 = Dist ** 2

    # Computation of Phi
    c_width2 = (factor * c_width) ** 2

    #rbfnet.Phi = np.exp(-Dist2 / (2 * c_width2))
    rbfnet_aux.Phi = np.exp(-Dist2 / (2 * c_width2))

    #return rbfnet
    return rbfnet_aux


def distances(G, S, T):
    '''
    Short-hand for MATLAB's distances function.
    '''

    cdist_All = dict(nx.shortest_path_length(G))

    grid = [(s[0], t[0]) for s in S for t in T]
    cdist = np.zeros(len(grid))
    for k, g in enumerate(grid):
        i, j = g[0], g[1]
        v = cdist_All.get(i)
        cdist[k] = v.get(j)

    return cdist.reshape((len(S), len(T)))


def Graph_EMesh(G, r, ldim):
    '''
    Graph_EMesh constructs an epsilon-mesh of size r on graph G.
    '''
    if r == 0:
        centres = np.array(range(len(G.nodes)))
        centres = centres.reshape((len(centres), 1))

        MInd = []

    else:
        deg = np.array([node[1] for node in G.degree])
        NId = np.where(deg > ldim)[0]

        Id1 = np.setdiff1d(np.array(range(len(G.nodes))), NId)

        N = np.array(range(len(G.nodes)))
        i = 0

        centres = []
        MInd = []
        while len(N) > 0:
            check = N[0] in NId
            if check == True:
                centres.append(N[0])
                MInd.append(list(nx.generators.ego_graph(G, N[0], radius=r, center=False)))
                MNodeInt = np.intersect1d(N, MInd[i])

                N = np.setdiff1d(N, N[0])
                N = np.setdiff1d(N, MNodeInt)

                i = i + 1

            else:
                N = np.setdiff1d(N, N[0])

    centres = np.array(centres).reshape((len(centres), 1))
    return centres, MInd

