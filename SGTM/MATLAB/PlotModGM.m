function [X,Y,Z,P] = PlotModGM(DataTot,TotMod,factor)

Int = max(DataTot) - min(DataTot);
bin_x = max(Int)/70;
x = min(DataTot(:,1)) - 5*bin_x:bin_x:max(DataTot(:,1)) + 5*bin_x;
y = min(DataTot(:,2)) - 5*bin_x:bin_x:max(DataTot(:,2)) + 5*bin_x;
z = min(DataTot(:,3)) - 5*bin_x:bin_x:max(DataTot(:,3)) + 5*bin_x;
[X,Y,Z] = meshgrid(x,y,z);
P = zeros(size(X));
for i=1:size(X,1)
% tic;
disp(i)
for j=1:size(X,2)
%         for k=1:size(X,3)
Data_Post = reshape([X(i,j,:) Y(i,j,:) Z(i,j,:)],3,size(X,3))';
P(i,j,:) = pdf(TotMod,Data_Post);
%         end
end
% toc;
end
% figure;
isovalue = max(max(max(P)))*factor;
IsoSurf = isosurface(X,Y,Z,P,isovalue);
p2 = patch(isosurface(X,Y,Z,P,isovalue));
isonormals(X,Y,Z,P,p2);
set(p2,'FaceColor',[1 0 0],'EdgeColor',[0.1 0.1 0.1]);
p2.FaceAlpha = 0.1 ;
p2.EdgeAlpha = 0.1;
drawnow



infile = open("Graph.pkl",'rb')
FG = pkl.load(infile)
FG = [FG]	
NData = np.loadtxt('D2.csv', delimiter=',')

Net, LogL, GMDist, NoisyMan = Standardized_AGTM_InitTrain(FG, NData, 2, 0.1, 0.1, 1)


#means, Cov = agtm_em(Net, 20, NData, 1)

#print(np.shape(d))




'''


infile = open("Graph.pkl",'rb')
FG = pkl.load(infile)
	
NData = np.loadtxt('D2.csv', delimiter=',')

Nodes = AllNodeNames(FG)
Net = gtminit(FG, Nodes, 2, 0.1, NData, 0.1)

means, Cov = agtm_em(Net, 20, NData, 1)

#print(np.shape(d))
'''


	
	
	
#This one also works but is slow
#distance_neighbors = np.zeros((len(S), len(T)))
#distance_list = []
#for i,v1 in enumerate(S):
#	for j,v2 in enumerate(T):
#		distance_neighbors[i,j] = nx.shortest_path_length(G, source=v1[0], target=v2[0])
#	
#print(distance_neighbors)	

'''
start = time.time()
grid = [(s[0], t[0]) for s in S for t in T]
D = np.zeros(len(grid))
for k,g in enumerate(grid):
i,j = g[0], g[1]
v = cdist.get(i)
D[k] = v.get(j)		
stop = time.time()
print(stop - start)
#print(D.reshape((len(S), len(T))))
'''	
	
'''	
infile = open("Graph.pkl",'rb')
FG = pkl.load(infile)
rbfnetnet = rbf(FG, 2, 0.1, 'Gaussian')
'''


'''
keys = cdist_All.keys()
i_coords, j_coords = np.meshgrid(S, T)
i_coords, j_coords = np.ravel(i_coords), np.ravel(j_coords)
	
cdist = np.zeros(len(i_coords))	
for k in range(len(i_coords)):
	i,j = i_coords[k], j_coords[k]
	v = cdist_All.get(i)
	cdist[k] = v.get(j)
		
'''












