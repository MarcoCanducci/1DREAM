function [centres,MInd] = Graph_EMesh(G,r,ldim) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
% Graph_EMesh constructs an epsilon-mesh of size r on graph G             %
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

if r==0
    centres = 1:size(G.Nodes,1);
    centres = centres';
    
else
    
    deg = degree(G);
    NId = find(deg>ldim);
    Id1 = setdiff(1:size(G.Nodes,1),NId);
    N = 1:size(G.Nodes);
    i = 1;
    while(isempty(N) == 0)
        if ismember(N(1),NId)
            centres(i,:) = N(1);
            MInd{i} = nearest(G,N(1),r);
            %disp(MInd)
            MNodeInt = intersect(N,MInd{i});
            %disp("here")
            %disp(MNodeInt)
            N = setdiff(N,N(1));
            N = setdiff(N,MNodeInt);
            
            i = i + 1;
        else
            N = setdiff(N,N(1));

        end
    end
end
