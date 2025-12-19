function [curv,mag,c_par,c_per,EndNodes,y,rbfnet] = Compute_CurvatureOn_rbf_FiniteDifference(G,a)

    Adj = adjacency(G);
    Abs_G = graph(Adj);
    rbfnet = rbf(G,2,1,a,'Gaussian');

    V_1 = str2num(char(G.Nodes.Name));
    [V_,mu,sigma] = zscore(V_1);
%     sigma = ones(1,3);
%     mu = zeros(1,3);
    Phi = rbfnet.Phi;
    H = Phi*Phi';
%     A1 = inv(H + 1e-2.*eye(size(H,1)));
    W = (H + 1e-1.*eye(size(H,1)))\Phi*V_;
%     W = Phi'\V_;
%     W = lsqminnorm(Phi',V_);
%     W = pinv(Phi')*V_;
    rbfnet.W = W;
    y = (W'*Phi)';
    y = y.*sigma + mu;
    rbfnet.V_ = y;
    eps_centres = rbfnet.c;
    
%     figure; hold on; 
%     gplot3(Adj,y); axis equal; hold on
%     plot3(y(eps_centres,1),y(eps_centres,2),y(eps_centres,3),'g.','markersize',10)
%     drawnow
    t = 0.5e-1;
%     w = [0.5-2*t ones(1,4).*t 0.5-2*t];
    w = zeros(2,6);
    w(1,:) = [ones(1,5).*t 1-5*t];
    w(2,:) =  flip(w(1,:));
    AddV = (1:5) + size(V_,1);
    
    EndNodes = Abs_G.Edges.EndNodes;
%     NNodes = [];
%     c_par = zeros(size(EndNodes,1),3);
%     c_per = c_par;
%     d1y = c_par;ar
%     d2y = c_par;
    mag = zeros(size(EndNodes,1),1);
    curv = mag;
    NNodes = [];
    
    for i=1:size(EndNodes,1)
%         disp(i)

        NewEdge = [EndNodes(i,1) AddV(1); AddV(1) AddV(2); AddV(2) AddV(3); ...
                    AddV(3) AddV(4); AddV(4) AddV(5); AddV(5) EndNodes(i,2)];
        
        
%         plot3(y(EndNodes(i,:)',1),y(EndNodes(i,:)',2),y(EndNodes(i,:)',3),'r.-')
        c_par = zeros(2,3);
        c_per = c_par;
        d1y = zeros(3,2);
        d2y = zeros(3,2);
        
        for j=1:2%size(w,1)

            GP = Abs_G;
            GP.Edges.Weight = ones(size(EndNodes,1),1);
%             GP = rmedge(GP,EndNodes(i,:));
            GP = addedge(GP,NewEdge(:,1),NewEdge(:,2),w(j,:));
    %         Adj2 = adjacency(GP);

    %         disp(EndNodes(i,:));
            yt = rbfmap_cont(GP,[EndNodes(i,1) AddV EndNodes(i,2)],rbfnet,mu,sigma);
            PD = pdist2(yt(:,1:end-1)',yt(:,1:end-1)');
            d_Norm(j) = sum(min(PD+1e10.*eye(size(yt,2)-1)));
            d_Norm(j) = d_Norm(j)./(5.*t);
    %         Nodes2 = [y; yt'];

    %         figure; gplot3(Adj2,Nodes2); hold on; axis equal

            yt = [y(EndNodes(i,1),:); yt'; y(EndNodes(i,2),:)]';
            plot3(y(EndNodes(i,:)',1),y(EndNodes(i,:)',2),y(EndNodes(i,:)',3),'r.-','markersize',5)
            plot3(yt(1,:),yt(2,:),yt(3,:),'k.-','markersize',5)

    %         pause(1)

            NNodes = [NNodes; yt'];

            d1y(:,j) = FirstDer_FD(GP,AddV(2:4),rbfnet,t,mu,sigma);
%             d1y(:,j) = d1y(:,j)./norm(d1y(:,j));
            d2y(:,j) = SecondDer_FD(GP,AddV(1:2:5),rbfnet,t,mu,sigma);
%             d2y(:,j) = d2y(:,j)./norm(d2y(:,j));
    %         quiver3(yt(1,3),yt(2,3),yt(3,3),d2y(i,1),d2y(i,2),d2y(i,3),'k');
%             drawnow

%             Proj_Op = d1y(:,j)*pinv(d1y(:,j));
%             c_par(j,:) =  d2y(:,j)'*Proj_Op;
%             c_per(j,:) = (eye(3,3) - Proj_Op)*d2y(:,j);
            
            
            
            
    %         
%             quiver3(yt(1,3),yt(2,3),yt(3,3),d1y(i,1),d1y(i,2),d1y(i,3),0,'g');
%             quiver3(yt(1,3),yt(2,3),yt(3,3),d2y(i,1),d2y(i,2),d2y(i,3),0,'b');
%             drawnow
        
        end
        
%         curv(i) = norm(c_per(i,:));
        curv(i) = nanmean(vecnorm(d2y'));%./mean(d_Norm);
        mag(i) = nanmean(vecnorm(c_par'));
        

        clearvars GP
%         drawnow
    end
%     curv = mean([curv mag],2);
%     plot3(NNodes(:,1),NNodes(:,2),NNodes(:,3),'k.','markersize',5)
%    drawnow
end

function yd1 = FirstDer_FD(G,m,net,h,mu,sigma)
    
    yt = rbfmap_cont(G,m,net,mu,sigma);
%     yt = yt.*sigma + mu;
    
    yd1 = 0.5.*(yt(:,3) - yt(:,1))./h;

end

function yd2 = SecondDer_FD(G,m,net,h,mu,sigma)
    
    yt = rbfmap_cont(G,m,net,mu,sigma);
    
%     yt = yt.*sigma + mu;
    
    yd2 = 0.25.*(yt(:,3) - 2.*yt(:,2) + yt(:,1))./(h^2);

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

