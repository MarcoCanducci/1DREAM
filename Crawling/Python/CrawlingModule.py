import numpy as np
import warnings
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore")


def OrderOnLengths(FG, NoisyMan):
    '''
    Order the graphs in both FG and NoisyMan on descending length.
    '''
    # calc lengths of the graphs
    distances = []
    for graph in FG:
        graphlength = 0
        for edge in graph.edges(data=True):
            graphlength += edge[2]['Distance']
        distances.append(graphlength)

    #sort FG on the lengths array
    FG = [x for _, x in sorted(zip(distances, FG), reverse=True)]
    NoisyMan = [x for _, x in sorted(zip(distances, NoisyMan), reverse=True)]

    return FG, NoisyMan



def Crawling_TwoPhases(D, r, ldim, betha):
	'''
	CRAWLING_TWOPHASES is a recursive procedure for building graphs locally
	aligned with the local tangent space of a manifold.
	
	Description:
	This function, given a manifold sampled in point cloud, recovers an 
	abstract and an embedded graph lying on the local tangent spaces of the
	manifold and describing its geometry.

	It is compose of three phases:
	- Initialiation: selects randomly a seed and computes the local tangent
	space of the manifold on that seed. It then identifies principal
	directions by performing decomposition of the local covariance matrix
	and estimates new candidate nodes.	
	-Expansion: given the set of nodes of degree 1 (having only 1 incident
	edge), it expands the crawling on the new locally linear principal
	directions, looking for new candidate nodes of the graph.
	- Contraction: Aims at contracting the set of new nodes by collapsing
	neighbouring nodes into one single node, given an accuracy threshold.

	Output:
	- Residual set R;
	- Embedded graph G (from which the abstract graph is extracted);
	'''
	# Initialization
	NEdges = ldim*2
	alpha = 0.75
	R = D

	G = nx.Graph()
	c = 0

			
	# Random initialization np.array([[ -2.677009,  -8.728987, -15.650824]])  #
	t0 = R[np.random.choice(R.shape[0], 1, replace=True)] 
	print("Seed Position: ", t0)
	N = rangesearch(R, t0, r)

	G.add_node(0, Name = t0.tobytes())
	Point_ID = 0

	# First iteration (no eigenvector projection)
	DN = R[N[0]]

	R = np.delete(R,np.where((R==t0).all(axis=1))[0], axis=0)
		
	pca = PCA()
	Fit = pca.fit_transform(DN)
	U = pca.components_.T
	

	if len(U[0,:])<ldim:
		return None, D, 0

	tNp = np.tile(t0, (ldim,1)) + (alpha*r)*U[:,0:ldim].T
	tNn = np.tile(t0, (ldim,1)) - (alpha*r)*U[:,0:ldim].T	



	tN = np.concatenate((tNp, tNn), axis=0)
	nbrs = NearestNeighbors(n_neighbors=1).fit(R)
	Dist, tNN_Idx = nbrs.kneighbors(tN)
	
	tNN = R[tNN_Idx.reshape((1,len(tNN_Idx)))]

	PDir = tNN - np.tile(t0, (NEdges,1))
	Mod = np.sqrt(np.sum(PDir[0]**2, axis=1))

	PDir = PDir[0] / np.tile(Mod.reshape((len(Mod),1)), (1,len(PDir[0][0,:])))
	m = len(PDir)
	tNN = np.unique(tNN, axis=1)[0]

	PDir_copy = PDir

	for i in range(len(tNN)):
		Nodes = AllNodeNames(G, len(t0[0]))

		P1 = PDir_copy[0,:]	
		PDir_copy = np.roll(PDir_copy, -1, axis=0)
		P2 = PDir_copy[0,:]

		ISM, ISM_ID = ismembertol(tNN[i,:], Nodes, 1e-4)

		G.add_node(Point_ID+1, Name = tNN[i,:].tobytes())
		R = np.delete(R,np.where((R==tNN[i,:]).all(axis=1))[0], axis=0) 
		Point_ID = Point_ID + 1

		e = (0, Point_ID) #Edge Tuple				
		G.add_edge(*e, Subspace_Basis = np.vstack((P1, P2)), Distance = Mod[i]) 

		Nodes = AllNodeNames(G, len(t0[0]))
		
	d = np.array([G.degree[k] for k in range(len(G.nodes))])	
	
	NewGen = []
	for i in range(NEdges):
		ind = np.where(d==(i+1)) # i+1 since counting degrees starts from 1 not 0
		NewGen.append(Nodes[ind, :][0]) 
	
	
	Nodes = AllNodeNames(G, len(t0[0]))
	A = len(Nodes)
	B = A - 1
	it = 1
		
	# Alternating Expansion and contraction phases
	while len(NewGen[0])>0 and len(R)>20 and B<A:	

		# Expansion Phase
		B = A
		for i in range(len(NewGen[0])):
			
			t0 = NewGen[0][i,:]
			t0 = t0.reshape((1, len(t0)))
			
			N = rangesearch(R,t0,r)
			if len(N[0]) <=5:
				
				# Border found
				RRR = R[N[0]]
				for k in range(len(RRR)):
					R = np.delete(R, np.where((R==RRR[k,:]).all(axis=1))[0], axis=0)
				
				continue

			DN = R[N[0]]
			pca = PCA()
			Fit = pca.fit_transform(DN)
			V = pca.components_.T			
			
			# Projection of parent subspace onto new eigenvectors
			NodeID = np.where((Nodes==t0).all(axis=1))[0][0]
			
			
			GN = list(G.adj[NodeID])[0]			
			S = G.edges[GN, NodeID]['Subspace_Basis']
			Sub = S.T

			
			P = np.dot( V[:, 0:ldim], np.linalg.pinv(V[:, 0:ldim]) )

			U = np.dot( P, Sub[:, 0:ldim] )
			for j in range(ldim):
				U[:, j] = U[:, j] / np.linalg.norm(U[:, j])
				
			tNp = np.tile(t0, (ldim,1)) + (alpha*r)*U[:, 0:ldim].T
			tNn = np.tile(t0, (ldim,1)) - (alpha*r)*U[:, 0:ldim].T
			
			tN = np.concatenate((tNp, tNn), axis=0)
									
			nbrs = NearestNeighbors(n_neighbors=1).fit(R)
			Dist, tNN_Idx = nbrs.kneighbors(tN)
	
			tNN = R[tNN_Idx.reshape((1,len(tNN_Idx)))]
			
			
			PDir = tNN - np.tile(t0, (NEdges,1))
			Mod = np.sqrt(np.sum(PDir[0]**2, axis=1))
			
			PDir = PDir[0] / np.tile(Mod.reshape((len(Mod),1)), (1,len(PDir[0][0,:])))
			
			m = len(PDir)
			PDir_copy = PDir			
			
			# Contraction Phase
			for j in range(len(tN)):

				tNNeigh = rangesearch(R, tNN[0][j,:].reshape((1, len(tNN[0][j,:]))), r)


				P1 = PDir_copy[0,:]	
				PDir_copy = np.roll(PDir_copy, -1, axis=0)
				P2 = PDir_copy[0,:]


				if len(tNNeigh[0]) < 5:

					continue 
											
				ISM, ISM_ID = ismembertol(tNN[0][j,:], Nodes, 1e-4)				


				if ISM == True:
					
					NNN = list(G.adj[ISM_ID])
					d = np.array([G.degree[k] for k in range(len(G.nodes))])
					if len(NNN) < NEdges and d[NodeID]< NEdges:

						x = Nodes[ISM_ID,:]
						x_ID = ISM_ID
						CreateEdge(G, x, x_ID, P2, m, j, NodeID, t0)
						
				
				
				else:
					RS = rangesearch(Nodes, tN[j,:].reshape((1, len(tN[j,:]))), betha*r)

			
					if len(RS[0]) != 0:

						RS1 = sorted(RS[0])
						NNN = list(G.adj[RS1[0]])
						d = np.array([G.degree[k] for k in range(len(G.nodes))])
						if len(NNN) < NEdges and d[NodeID]< NEdges:

							x = Nodes[RS1[0]]						
							x_ID = RS1[0]

							CreateEdge(G, x, x_ID, P2, m, j, NodeID, t0)
					
					else:
						# Create new node and connect to t0
						NNN = list(G.adj[NodeID])
						d = np.array([G.degree[k] for k in range(len(G.nodes))])
						if len(NNN) < NEdges and d[NodeID]< NEdges:
							
							G.add_node(Point_ID+1, Name = tNN[0][j,:].tobytes())

							R = np.delete(R,np.where((R==tNN[0][j,:]).all(axis=1))[0], axis=0)

							Point_ID = Point_ID + 1
							
							e = (NodeID, Point_ID) 
							G.add_edge(*e, Subspace_Basis = np.vstack((P1, P2)), Distance = Mod[j])
					
						
	
				Nodes = AllNodeNames(G, len(t0[0]))

	
	
		for tup in G.edges:
			if tup[0] == tup[1]:
				G.remove_edge(tup[0], tup[1])

		d = np.array([G.degree[k] for k in range(len(G.nodes))])	
		Nodes = AllNodeNames(G, len(t0[0]))
		

		# Updating the set of 1-edge nodes		
		NewGen = []
		for i in range(NEdges):
			ind = np.where(d==(i+1))
			NewGen.append(Nodes[ind, :][0])
	
		
		if len(NewGen[NEdges-1])>0:

			R4 = rangesearch(R, NewGen[NEdges-1], betha*r)
			R2 = R
		
			for j in range(len(NewGen[NEdges-1])):
				if len(R4[j])>0:
					RR = R[R4[j]]
					for k in range(len(RR)):
						R2 = np.delete(R2, np.where((R2==RR[k,:]).all(axis=1))[0], axis=0)
		 	
			R = R2
			
		# Updating size of nodes for conditional break		
		A = len(Nodes)
		it = it + 1
		print("it", it)
		
		
									
	return G, R, c	


def AllNodeNames(G, Lt0):
    '''
    Stacks all Node names into an nd-array
    '''

    Nodes = list(G.nodes(data=True))

    Names = np.zeros((len(Nodes), Lt0))
    for i, n in enumerate(G.nodes):
        Names[i, :] = np.frombuffer(G.nodes[n]['Name'])
    return Names


def ismembertol(check, X, tol):
    '''
    Short-hand for MATLAB ismembertol function
    '''

    X_checked = np.isclose(check, X, rtol=tol).all(axis=1)
    ISM = True in X_checked
    if ISM == True:
        ISM_ID = np.where(X_checked == True)[0][0]
    else:
        ISM_ID = []
    return ISM, ISM_ID


def CreateEdge(G, x, x_ID, P2, m, j, NodeID, t0):
    RecE = G.has_edge(NodeID, x_ID)
    if RecE == False:
        Dir = x - t0
        Mod2 = np.linalg.norm(Dir)
        Dir = Dir / Mod2

        e = (NodeID, x_ID)  # Edge Tuple
        G.add_edge(*e, Subspace_Basis=np.vstack((Dir, P2)), Distance=Mod2)


def MultiM(D, NoisyD, r, ldim, betha):
	'''
	MULTIM is a recursive procedure for trimming graphs recovered through
	crawling and recovering of the noisy sampled manifold.
	
	Description:
	This function, given a point cloud sampling multiple noisy manifolds
	and a data set containing the same manifolds but collapsed onto theri
	respective spine, recovers a set of graphs and a set of point clouds
	describing the geometry of the manifolds and the noisy distributions of
	points surrounding them. It applies Manifold crawling iteratively until
	the residual set's size is minimized.

	See also: Crawling_TwoPhases

	'''
	TotR = D
	it = 1
	rN = r
	
	CollG = []
	

	while len(TotR) > 20:
	
		G, TotR, c = Crawling_TwoPhases(TotR, r, ldim, betha)
		
		if len(TotR) == 0:
			#print("Remaining number of points: ", len(TotR))
			CollG.append(G)
			break
		
		if c == 0 and G != None:
			Nodes = AllNodeNames(G, 3)
			
			if len(Nodes) > ldim*2 + 1:
				#print("Remaining number of points: ", len(TotR))
				NN = rangesearch(TotR, Nodes, rN)
				NN = np.unique(np.hstack(NN))
				
				TotR = TotR[np.setdiff1d(range(len(TotR)), NN)]

				Dist = [G.edges[tup[0], tup[1]]['Distance'] for tup in G.edges]
				
				CollG.append(G)
				
				it = it + 1	
			
			
	k = 1
	NoisyMan = []
	FG = []
	for i in range(len(CollG)):
		
		components = np.unique(list(nx.connected_components(CollG[i])))

		if len(components) > 1:

			for j in range(len(components)):

				SubGraph = CollG[i].subgraph(components[j])
				Nodes = AllNodeNames(SubGraph, 3)
				if len(Nodes) > ldim*2 + 1:

					NN = rangesearch(NoisyD, Nodes, rN)
					NN = NoisyD[np.unique(np.hstack(NN)), :]
					NoisyMan.append(NN)
					
					FG.append(SubGraph)
					
					k = k + 1
					
		else:
			SubGraph = CollG[i]
			Nodes = AllNodeNames(SubGraph, 3)
			if len(Nodes) > ldim*2 + 1:
				NN = rangesearch(NoisyD, Nodes, rN)
				NN = NoisyD[np.unique(np.hstack(NN)), :]
				NoisyMan.append(NN)
				
				FG.append(SubGraph)
				
				k = k + 1

				
	return FG, NoisyMan, TotR			


def rangesearch(X, Y, Radius, return_distances=False):
    '''
    - Short-hand for applying MATLAB's rangesearch function
    for finding indices and distances to nearest neighbors (NN).
    - Default for returning distances is False.
    '''

    tree = KDTree(X)
    NN = tree.query_radius(Y, Radius, return_distances)

    return NN
