function [B,K,M,z] = mf2d_cvx(Y,Tu,Ty,Tth,TH,lambda1,lambda2,s,nu,ny,nth)
    cvx_begin quiet
        variables B(nu*s,ny) K(ny*s,ny) M(nth*ny*s,ny) z(nth,1)
        minimize( sum(sum_square(Y-[Tu,Ty,-Tth,TH]*[B;K;M;kron(eye(ny),z)])) + ...
            lambda1*norm_nuc([reshape(permute(reshape(M,ny*nth,s,ny),[1,3,2]),ny*nth,s*ny)',...
                reshape(permute(reshape(K,ny,s,ny),[1,3,2]),ny,s*ny)';
                kron(eye(ny),z'), eye(ny)]) + ...
            lambda2*norm(z,1) )
    cvx_end
end