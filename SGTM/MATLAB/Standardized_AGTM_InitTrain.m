function Standardized_AGTM_InitTrain(FG,F_NoisyData,IntDim,r,epsilon,mem)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
% Standardized_AGTM_InitTrain sets up AGTM and performs training calling  %
% AGTM_EM after standardization of datasets and graphs                    %
%                                                                         %
%  Copyright (C) 2020  Marco Canducci                                     %
%  Email: marco.canducci91@gmail.com                                      %
%                                                                         %
%  This program is free software: you can redistribute it and/or modify   %
%  it under the terms of the GNU Affero General Public License as         %
%  published by the Free Software Foundation, either version 3 of the     %
%  License, or (at your option) any later version.                        %
%                                                                         %
%  This program is distributed in the hope that it will be useful,        %
%  but WITHOUT ANY WARRANTY; without even the implied warranty of         %
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          %
%  GNU Affero General Public License for more details.                    %
%                                                                         %
% You should have received a copy of the GNU Affero General Public License%
% along with this program.  If not, see <https://www.gnu.org/licenses/>.  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%figure; hold on
F_NoisyData = double(F_NoisyData);
k =1;
for i=1:size(FG,2)  
    %tic;
    %disp(i)
    disp(['net n. ', int2str(i)]);
    Adj = adjacency(FG{i});    
    Nodes = str2num(char(FG{i}.Nodes.Name));
    NN = rangesearch(F_NoisyData,Nodes,r);
    T = F_NoisyData(unique(cell2mat(NN')),:);

    G = graph(Adj);
    G.Nodes.Name = num2cell(num2str(Nodes),2);
    

    r_AGTM = mean(min(pdist2(Nodes,Nodes)+1e10.*eye(size(Nodes,1))));
    net{k} = gtminit(G,IntDim,epsilon,T,r);
    disp(size(T))
    
    %Stop here 
    
    %[net{k},logL{k}] = agtm_em(net{k},20,T,mem);    

    %priors = net{k}.gmmnet.GMdist.ComponentProportion;
    %GMDist{k} = gmdistribution(net{k}.gmmnet.V_,net{k}.gmmnet.Sigma,priors);
    %NoisyMan{k} = T;
    %k = k+1;
    

end