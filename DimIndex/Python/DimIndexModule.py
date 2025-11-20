import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree

def seperateDimensions(F_Data, DN, Idx):
    '''
    Seperates the
    '''
    D = [F_Data[np.where(Idx == 0)], F_Data[np.where(Idx == 1)], F_Data[np.where(Idx == 2)]]
    DN = [DN[np.where(Idx == 0)], DN[np.where(Idx == 1)], DN[np.where(Idx == 2)]]

    return D, DN

def loadData(filename1, filename2):
    '''
    Loads data from csv file with filename1 and filename2 with the delimiter=','
    '''
    DTot = np.loadtxt(filename1, delimiter=',')
    SAF_D = np.loadtxt(filename2, delimiter=',')
    return DTot, SAF_D

def CartesiantoBarycentric(tri, points):
    '''
    -(lambda1, lambda2, lambda3) are the new Barycentric Coordinates
     for a point such that lambda3 = 1 - lambda1 - lambda2.
    -Refer to https://en.wikipedia.org/wiki/Barycentric_coordinate_system

    -tri: triangle defined by three vertex points.
    -points: points in tri to be converted.
    '''

    x1, y1, z1 = tri[0, 0], tri[0, 1], tri[0, 2]
    x2, y2, z2 = tri[1, 0], tri[1, 1], tri[1, 2]
    x3, y3, z3 = tri[2, 0], tri[2, 1], tri[2, 2]

    T = np.array([[x1 - x3, x2 - x3], [y1 - y3, y2 - y3]])
    DetT = np.linalg.det(T)

    lambda1 = (y2 - y3) * (points[:, 0] - x3) + (x3 - x2) * (points[:, 1] - y3)
    lambda1 = lambda1 / (DetT)

    lambda2 = (y3 - y1) * (points[:, 0] - x3) + (x1 - x3) * (points[:, 1] - y3)
    lambda2 = lambda2 / (DetT)

    lambda3 = 1 - lambda1 - lambda2

    return np.column_stack((lambda1, lambda2, lambda3))


def Dim_Index(F_Data, r, Simplex, smooth):
    '''
    Dim_Index, assigns to every point in a data set a dimensionality index.
    '''

    # Find rescaled eigenvectors using barycentric cohordinates (B) and
    # denoised eigenvalues.
    r_Idx = r
    NewData = F_Data
    B, L = Filtered_Dim_Est(F_Data, r_Idx)
    Struct.Eig = L
    Struct.Baryc_Eig = B
    #idx = np.zeros((len(F_Data), 4))
    idx = np.zeros((len(F_Data), 2))

    if Simplex == "Original":

        # Definition of Original simplex (portion) vertices and equidistant point.
        Eig = np.abs(L)
        V = np.array([[1, 0, 0], [0.5, 0.5, 0], [1. / 3, 1. / 3, 1. / 3]])
        m0 = np.mean(V, axis=0)

        # Constraints are:
        # x1 + x2 + x3 = 1
        # x1 >= x2 >= x3 >= 0
        cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1},
                {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},
                {'type': 'ineq', 'fun': lambda x: x[1] - x[2]},
                {'type': 'ineq', 'fun': lambda x: x[2]})

        res = minimize(ErrF, m0, method='SLSQP', constraints=cons, tol=1e-6)
        m = res.x.reshape((1, len(res.x)))

    elif Simplex == "Barycentric":

        # Definition of whole simplex vertices and equidistant points.
        Eig = np.abs(B)
        V = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        m = np.mean(V, axis=0).reshape((1, 3))

    # First index: kmedoids on eigenvalues using vertices as propototypes.

    # Index as smallest distance from vertex
    print("Computing second Index")
    scale = cdist(V, m, FisherMetricDist)
    Dist = cdist(Eig, V, FisherMetricDist)

    ID2 = np.argmin(Dist, axis=1)
    idx[:, 0] = ID2

    Struct.Idx1 = idx[:, 0]
    Struct.Idx2 = np.zeros(len(idx))  # idx[:,1]

    # l1 and l2 kernels computation.
    print("Computing third and fourth Index")
    K1 = np.exp(-Dist / (2 * scale[0]))
    K2 = np.exp(-(Dist ** 2) / (2 * scale[0] ** 2))

    K11 = np.sum(K1, axis=1)
    K1 = K1 / np.tile(K11.reshape((len(K11), 1)), (1, len(K1[0, :])))
    K22 = np.sum(K2, axis=1)
    K2 = K2 / np.tile(K22.reshape((len(K22), 1)), (1, len(K1[0, :])))

    # Entropies computation
    log3P1 = np.log(K1) / np.log(3)
    H1 = - np.sum(K1 * log3P1, axis=1)
    log3P2 = np.log(K2) / np.log(3)
    H2 = - np.sum(K2 * log3P2, axis=1)

    T1 = 1

    DNewH1 = NewData[np.where(H1 <= T1)]
    EigH1 = Eig[np.where(H1 <= T1)]
    H1_Idx = np.argmax(K1[np.where(H1 <= T1)], axis=1)

    #idx[:, 1] = H1_Idx
    Struct.Kernel1_Data = DNewH1
    Struct.Kernel1_Idx = H1_Idx

    T2 = 1

    DNewH2 = NewData[np.where(H2 <= T2)]
    EigH2 = Eig[np.where(H2 <= T2)]
    H2_Idx = np.argmax(K2[np.where(H2 <= T2)], axis=1)

    #idx[:, 2] = H2_Idx
    Struct.Kernel1_Data = DNewH2
    Struct.Kernel1_Idx = H2_Idx

    # Smoothing kernels on data space
    print("Computing fifth Index")
    if smooth == "l1":
        K = K1
    elif smooth == "l2":
        K = K2

    r_smooth = 2.0 * r	#Smoothing kernel is 2.0, r = 10e-16 we obtain the same result the original index 0.
    K_scale2 = 2.0 * (r_smooth ** 2)
    NN_search = rangesearch(NewData, NewData, r_smooth, return_distances=True)
    NN = NN_search[0]
    Dist = NN_search[1]
    S = np.zeros(np.shape(K))

    for i in range(len(NN)):
        W = np.exp(-(Dist[i] ** 2) / K_scale2)
        W2 = np.tile(W.reshape((len(W), 1)), (1, 3)) * K[NN[i], :]
        S[i, :] = np.sum(W2, axis=0) / np.sum(W)

    log3S = np.log(S) / np.log(3)
    HS = -np.sum(S * log3S, axis=1)

    TS = 2

    DNewS = NewData[np.where(HS <= TS)]
    EigS = Eig[np.where(HS <= TS)]
    S_Idx = np.argmax(S[np.where(HS <= TS)], axis=1)

    #idx[:, 3] = S_Idx
    idx[:, 1] = S_Idx
    Struct.K_Smooth_Data = DNewS
    Struct.K_Smooth_Idx = S_Idx

    return Struct, idx


class Struct:
    '''
     Dimentionality index struct
    '''

    def __init__(self, Eig, Baryc_Eig, Idx1, Idx2, Kernel1_Data, Kernel1_Idx, K_Smooth_Data, K_Smooth_Idx):
        self.Eig = Eig
        self.Baryc_Eig = Baryc_Eig
        self.Idx1 = Idx1
        self.Idx2 = Idx2
        self.Kernel1_Data = Kernel1_Data
        self.Kernel1_Idx = Kernel1_Idx
        self.K_Smooth_Data = K_Smooth_Data
        self.K_Smooth_Idx = K_Smooth_Idx


def ErrF(x):
    N = len(x)
    x = x.reshape((1, N))
    print(x)
    v1 = np.array([[1, 0, 0]])
    v2 = np.array([[0.5, 0.5, 0]])
    v3 = np.array([[1. / 3, 1. / 3, 1. / 3]])

    F = (cdist(x, v1, FisherMetricDist) - cdist(x, v2, FisherMetricDist)) ** 2 + \
        (cdist(x, v2, FisherMetricDist) - cdist(x, v3, FisherMetricDist)) ** 2 + \
        (cdist(x, v1, FisherMetricDist) - cdist(x, v2, FisherMetricDist)) ** 2

    return F[0]


def Filtered_Dim_Est(Data, Radius):
    '''
    FILTERED_DIM_EST_NEW determines the intrinsic dimensionality of a cloud
    of points.


    Description

    Input parameters:
        -   Data_Original: Data not yet processed using SAF.
        -   Data: Matrix Nx3 containing the observed points in a 3D space.
                 N is the number of observations, 3 is the dimensionality
                 of the embedding space.
        -   Radius: Radius for the neighbouring search.

    Output parameters:
        -   LNew: Matrix containing eigenvalues associated to each
            obs' neighbourhood, renormalized by their sum.
        -   NewLabels: Logic array containing "1" for the valid obs and
           "0" for the rejected ones.
        -   NewData: Matrix containing the valid obs of which a
            dimensionality index has been found.
        -   idx: column vector with dimensionality index for each valid
            observation.
        -   C: 3x3 matrix with the new centres after kmedoids.
        -   DistM: N_Valid_obs x 3 matrix with the distances of each valid
            obs to the initial clustering centres.


        This function can be modified to also return the following variables.
        -LNew: the matrix containing the renormalized eigenvectors.
        -NewLabels: the labels for the selection of valid obs.
        -NewData: the valid obs
        -idx: the dimensionality index associated to each obs in Data.
        -C: the centres after the clustering through k-medoids is performed.
        -DistM: the matrix with all distances of the obs to the centres as
                computed with FisherMetricDist.


    For each obs in Data, the covariance matrix of its neighbourhood is
    decomposed into its eigenvalues. These get then normalized by their sum,
    so that each triade of eigenvalues lies on a portion of the
    3d-simplex. A further step is applied for distributing the eigenvalues
    in the complete simplex ([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) so that
    the its mean is equidistant from each vertex and coincides with the point
    [1/3, 1/3, 1/3]. Here probabilities vary between [0 1] for each one of the
    dimensions. The distance of each triade of new rescaled eigenvaòues
    to the vertices is computed and stored in DistM, while the MATLAB
    implementation of the algorithm k-medoids is used for clustering obs
    with respect to the centres (used as clusters' prototypes).

    '''

    Neighbors2 = rangesearch(Data, Data, Radius)
    LNew = np.zeros(np.shape(Data))

    pca = PCA()

    k = 1
    C = []
    for i in range(len(Data)):
        '''
        By evaluating neighborhoods in the original dataset we can discard all
        points which didn't have neighbors populated enough to survive before
        appyling SAF. This step removes artifacts of the SAF algorithm.
        '''
        Fit = pca.fit_transform(Data[Neighbors2[i]])
        Lambda2 = pca.explained_variance_
        Lambda2 = sorted(Lambda2, reverse=True)

        LambdaTot = np.zeros(len(Data[0, :]))
        LambdaTot[:len(Lambda2)] = Lambda2
        LambdaTot = LambdaTot / np.sum(LambdaTot)
        LNew[i, :] = LambdaTot

        if any(np.isnan(LambdaTot)):
            print(i)
            C.append(i)
            k += 1

    # Rescaling of the eigenvalues using the barycentrc coordinates with respect to
    # the portion of simplex of vertices (1, 0, 0), (0.5, 0.5, 0), (1/3, 1/3, 1/3).

    tri = np.array([[1, 0, 0], [0.5, 0.5, 0], [1. / 3, 1. / 3, 1. / 3]])
    B = CartesiantoBarycentric(tri, LNew)

    return B, LNew


def Filtering(Data_Original, Data, Radius, cutoff=5):
    '''
    -Filtering, given a noisy data set and a diffused one (obtained by SAF)
     filters out noisy data points.
    -Data_Original: Orginal set with background noise
    -Data: result after applying SAF
    '''

    Neighbors1 = rangesearch(Data_Original, Data_Original, Radius)
    Neighbors2 = rangesearch(Data, Data, Radius)
    Labels = np.zeros(len(Data))

    for i in range(len(Data)):
        '''
        By evaluating neighborhoods in the original dataset we can discardall 
        points which didn't have neighbors populated enough to survive before
        appyling SAF. This step removes artifacts of the SAF algorithm.
        '''
        if Labels[i] == 0:
            if len(Neighbors2[i]) <= cutoff or len(Neighbors1[i]) <= cutoff:
                continue
            else:
                Labels[Neighbors2[i]] = 1


    return Data_Original[np.where(Labels==1)], Data[np.where(Labels == 1)], Labels, Neighbors1, Neighbors2


'''
 FisherMetricDist computes the geodesic distance on simplex between any
 two set of points XI and XJ.                                          

   Description

   Input parameters:
       -   XI: Matrix of objective observations w.r.t. which compute the
           distances.
       -   XJ: Matrix of the query obs.

   Output parameters:
       -   Distance matrix containing the distance to each objective obs
           in XI of the query obs in XJ.

   The geodesic distances on the d-simplex are computed as the shortest
   curves onto the positive portion of the d-sphere of radius 2:

   d(x1,x2) = 2*arccos(sum(sqrt(x1*x1')))
'''

FisherMetricDist = lambda XI, XJ: np.arccos(np.dot(np.sqrt(XJ), np.sqrt(XI)))


def rangesearch(X, Y, Radius, return_distances=False):
    '''
    - Short-hand for applying MATLAB's rangesearch function
    for finding indices and distances to nearest neighbors (NN).
    - Default for returning distances is False.
    '''

    tree = KDTree(X)
    NN = tree.query_radius(Y, Radius, return_distances)

    return NN
