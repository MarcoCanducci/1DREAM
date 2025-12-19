function [curv,EndNodes,y,rbfnet] = Compute_CurvatureOn_rbf_FiniteDifference_NewVersion_Nodes(G,a,c)

    Adj = adjacency(G);
    Abs_G = graph(Adj);
    rbfnet = rbf(G,2,1,a,'Gaussian');

    V_1 = str2num(char(G.Nodes.Name));
    [V_,mu,sigma] = zscore(V_1);
    Phi = rbfnet.Phi;
    H = Phi*Phi';

    W = (H + 1e-1.*eye(size(H,1)))\Phi*V_;
    rbfnet.W = W;
    y = (W'*Phi)';
    y = y.*sigma + mu;
    rbfnet.V_ = y;
    
    t = 1e-2;
    
    w = zeros(2,6);
    w(1,:) = [ones(1,5).*t 1-5*t];
    w(2,:) =  flip(w(1,:));
    AddV = (1:5) + size(V_,1);
    
%     w = zeros(2,5);
%     w(1,:) = [ones(1,4).*t 1-5*t];
%     w(2,:) =  flip(w(1,:));
%     AddV = (1:4) + size(V_,1);
    
    EndNodes = Abs_G.Edges.EndNodes;

    curv = zeros(size(EndNodes,1),1);
    
    for i=1:size(EndNodes,1)
%         disp(i)

        NewEdge = [EndNodes(i,1) AddV(1); AddV(1) AddV(2); AddV(2) AddV(3); ...
                    AddV(3) AddV(4); AddV(4) AddV(5); AddV(5) EndNodes(i,2)];
        
        for j=1:size(w,1)

            GP = Abs_G;
            GP.Edges.Weight = ones(size(EndNodes,1),1);
            GP = addedge(GP,NewEdge(:,1),NewEdge(:,2),w(j,:));

            yt = rbfmap_cont(GP,AddV,rbfnet,mu,sigma);
            yt = [y(EndNodes(i,1),:); yt'; y(EndNodes(i,2),:)];
            
%             yt = [y(EndNodes(i,j),:); yt'];
%             t1 = yt(3,:) - yt(1,:);
%             t1 = t1./norm(t1);
%             t2 = yt(5,:) - yt(3,:);
%             t2 = t2./norm(t1);   
%             
%             PD = min(pdist2(yt(2:end,:),yt(2:end,:))+ 1e10.*eye(size(yt,1)-1));
%             
%             
%             per_Vec = (t2-t1)./sum(PD(2:3));
            h_t = c*t;
            T_1 = 0.5.*(yt(3,:) - yt(1,:))./h_t;
            
            if j==1
%             h_t = mean(min(pdist2(yt,yt) + 1e10.*eye(size(yt,1))));
                G = (yt(5,:) - 2.*yt(3,:) + yt(1,:))./(4.*(h_t^2));
            else
                G = (yt(2,:) - 2.*yt(4,:) + yt(6,:))./(4.*(h_t^2));
            end
                % Tangent space in x2
            
%             T_sp_x2 = (yt(4,:) - yt(2,:));
%             T_sp_x2 = T_sp_x2'./norm(T_sp_x2);
            
%             Proj_Op = T_sp_x2*pinv(T_sp_x2);

%             gamma(j) = norm(G);
%             per_Vec = (eye(3,3) - Proj_Op)*G';
            gamma(j) = norm(G);        
        end
        curv(i) = mean(gamma);        

        clearvars GP
    end
end

function yt = rbfmap_cont(G,m,net,mu,sigma)
    
    eps_centers = net.c;
    c_width = net.c_width;
    W = net.W;

    dist = distances(G,eps_centers,m);
    arg_exp = -dist.^2./(2*(c_width^2));
    
    yt = (exp(arg_exp)'*W)';
    
    yt = (yt'.*sigma + mu)';
%     yt = yt';
end

