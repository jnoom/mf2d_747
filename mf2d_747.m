% Instead of the real flight data, the longitudinal dynamics of a Boeing 747 (Bryson 1994) are simulated.
% The airspeed and climb rate are measured while the elevator deflection is
% used as input.

clear all
addpath('functions')
rng(2024)

%% Time variables
ts = 0.1;  % sampling time
tf = 100;  % final simulation time
t  = ts:ts:tf;
N  = length(t)-1;

%% Variables for model-free fault diagnosis algorithm
s       = 4;    % VARX order
lambda1 = 0.1;  % tuning parameter for nuclear norm
lambda2 = 0.2;  % tuning parameter for 1-norm
nu      = 1;    % no. inputs
ny      = 2;    % no. outputs

%% Dictionary
nf = 40;  % number of possible fault instances per sensor

th = zeros(ny,nf,N+1);
for i=1:nf
	th(1,i,i*round(N/nf+1):end) = 1;
end
for i=1:nf
    th(2,nf+i,i*round(N/nf+1):end) = 1;
end

%% Model of Boeing 747 cruising at 40000 ft altitude and velocity 774 ft/s (units in ft, s, crad) (Bryson 1994)
Afc  = [-0.003 0.039 0 -0.322;
        -0.065 -0.319 7.74 0;
        0.0201 -0.101 -0.429 0;
        0 0 1 0];
Bfc  = [0.01; -0.18; -1.16; 0];
Cfc  = [1 0 0 0; 0 -1 0 7.74];
Dfc  = zeros(2,1);
sysf = c2d(ss(Afc,Bfc,Cfc,Dfc),ts);

A=sysf.A; B=sysf.B; C=sysf.C; D=sysf.D;
n = length(A(1,:));  % state-space model order

%% Noise
R = 10^-3*eye(ny);  % measurement noise
Q = 10^-8*eye(n);   % process noise

%% Input (elevator deflection)
u = repmat(0.5*[zeros(1, round(10/ts)), -ones(1, round(6/ts)), ones(1, round(4/ts)),...
    -ones(1, round(2/ts)), ones(1, round(2/ts)), zeros(1, round(6/ts)),...
    ones(1, round(6/ts)), -ones(1, round(4/ts)), ones(1, round(2/ts)),...
    -ones(1, round(2/ts)), zeros(1, round(6/ts)) ],1,2);

%% Output sequence
x(:,1) = zeros(n,1);
for k = 1:N+1
    x(:,k+1) = A*x(:,k) + B(:,1)*u(k) + sqrt(Q)*randn(n,1);
    y(:,k)   = C*x(:,k) + D(:,1)*u(k) + sqrt(R)*randn(ny,1);
end

%% Fault signals
z = zeros(nf*2,1);
nzi = [round(nf/2),round(nf/1.5),nf+round(nf/1.3),nf+round(nf/1.1)];  % indices of nonzeros in z
z(nzi(1)) = 1;
z(nzi(2)) = -1;
z(nzi(3)) = 1;
z(nzi(4)) = -1;

for k=1:N+1
    d(:,k) = th(:,:,k)*z; 
end
y = y+d;

%% Model-free fault diagnosis
% Construct data matrices
for i=1:N+1
    uv(i,:)  = {reshape(u(:,i),[1,length(u(:,1))])};
    yv(i,:)  = {reshape(y(:,i),[1,length(y(:,1))])};
    thv(i,:) = {reshape(th(:,:,i)',[1,numel(th(:,:,1))])};
end
Tth = cell2mat(thv(toeplitz(s:N,s:-1:1)));
Tu  = cell2mat(uv(toeplitz(s:N,s:-1:1)));
Ty  = cell2mat(yv(toeplitz(s:N,s:-1:1)));
TH  = cell2mat(thv(s+1:N+1));

% Perform optimization
[Bh,Kh,Mh,zh] = mf2d_cvx(y(:,s+1:N+1)',Tu,Ty,Tth,TH,lambda1,lambda2,s,nu,ny,nf*2);

%% Validation
for k=1:N+1
    dh(:,k)  = th(:,:,k)*zh;  % reconstruct sensor faults
end
for k = s+1:N+1
    yh(:,k) = Bh'*vec(u(:,k-1:-1:k-s)) + Kh'*vec(y(:,k-1:-1:k-s)) - Kh'*vec(dh(:,k-1:-1:k-s)) + dh(:,k);
end
VAF = vaf(y(:,s+1:N+1)',yh(:,s+1:N+1)');

%% Figures
figure;
subplot(211); plot(t,d(1,:),'Color',"#D95319",'linewidth',2);hold on;
plot(t,dh(1,:),'Color',"#EDB120",'linestyle',':','linewidth',3);
ylabel('$f_{V}$ (ft/s)','fontsize',14,'interpreter','latex');grid; set(gca,'fontsize',14);
h2=legend('True','MF2D'); set(h2,'color','white','edgecolor','black','interpreter','latex');
subplot(212); plot(t,d(2,:),'Color',"#D95319",'linewidth',2);hold on;
plot(t,dh(2,:),'Color',"#EDB120",'linestyle',':','linewidth',3);
ylabel('$f_{h}$ (crad)','fontsize',14,'interpreter','latex');grid; set(gca,'fontsize',14);
xlabel('Time (s)','fontsize',14,'interpreter','latex')

figure;
subplot(311);plot(t,y(1,:),'linewidth',2);ylim([-4, 5])
ylabel('$V_{TAS}$ (ft/s)','fontsize',14,'interpreter','latex');grid; set(gca,'fontsize',14);
subplot(312);plot(t,y(2,:),'linewidth',2);ylim([-15, 15])
ylabel('$\dot{h}$ (ft/s)','fontsize',14,'interpreter','latex');grid; set(gca,'fontsize',14);
subplot(313);plot(t,u,'linewidth',2);ylim([-1 1])
ylabel('$\delta_e$ (crad)','fontsize',14,'interpreter','latex');grid; set(gca,'fontsize',14);
xlabel('Time (s)','fontsize',14,'interpreter','latex')