clear;
clc;

xmin=-1.6;
xmax=0;
ymin=-0.8;
ymax=0.8;
zmin=-0.8;
zmax=0.8;

global rou
rou=0.3;

% syms x1f x2f x3f p1f p2f p3f dab1 dab2 dab3 sigma beta rou kapa miu
% f1=sigma*(x2f-x1f);
% f2=rou*x1f-x2f-x1f*x3f;
% f3=-beta*x3f+x1f*x2f+(kapa-0.5)*miu*x3f;
% f1x1=gradient(f1,x1f);
% f2x2=gradient(f2,x2f);
% f3x3=gradient(f3,x3f);
% H=1/4*(p1f^2+p2f^2+(1+miu*x3f^2)*p3f^2)+(f1-dab1)*p1f+(f2-dab2)*p2f+(f3-dab3)*p3f-f1x1-f2x2-f3x3+miu*x3f*f3/(1+miu*x3f^2)+dab1^2+dab2^2+dab3^2/(1+miu*x3f^2);
% Hx1=gradient(H,x1f);
% Hx2=gradient(H,x2f);
% Hx3=gradient(H,x3f);
% Hp1=gradient(H,p1f);
% Hp2=gradient(H,p2f);
% Hp3=gradient(H,p3f);

xnode=[-1;0;0];

Nphi=400;            % »·ÉÏ»®·Ö¾«¶È
Ntheta=400;
phi=linspace(0,pi,Nphi);
theta=linspace(0,2*pi,Ntheta);
[PHI,THETA]=meshgrid(phi,theta);
PHI=reshape(PHI,1,Nphi*Ntheta);
THETA=reshape(THETA,1,Nphi*Ntheta);

A=[-2*(3*xnode(1)^2-1), -rou, -rou;
    2*rou*(3*xnode(1)^2-1), -1, 0;
    2*rou*(3*xnode(1)^2-1), 0, -1;];
B=zeros(6,6);
B(1:3,1:3)=A;
B(1:3,4:6)=[1 0 0;0 1 0;0 0 1];
B(4:6,4:6)=-A';
[Bv,Be]=eig(B);
Bv1=[];
Bv2=[];
for i=1:6
    if Be(i,i)>0
        Bv1=[Bv1 Bv(1:3,i)];
        Bv2=[Bv2 Bv(4:6,i)];
    end
end
M=real(Bv2/Bv1);

R=1e-2;
tf=5;
h=0.005;
nT=tf/h;
Np=zeros(1,Nphi*Ntheta);
xlamS=zeros(7,Nphi*Ntheta);
xlamS(1:3,:)=[xnode(1)+R*sin(PHI).*cos(THETA);xnode(2)+R*sin(PHI).*sin(THETA);xnode(3)+R*cos(PHI)];
xlamS(4:6,:)=M*[R*sin(PHI).*cos(THETA);R*sin(PHI).*sin(THETA);R*cos(PHI)];
xlamS(7,:)=1/2*(M(1,1)*(xlamS(1,:)-xnode(1)).^2+M(2,2)*(xlamS(2,:)-xnode(2)).^2+M(3,3)*(xlamS(3,:)-xnode(3)).^2+...
    2*M(1,2)*(xlamS(1,:)-xnode(1)).*(xlamS(2,:)-xnode(2))+2*M(1,3)*(xlamS(1,:)-xnode(1)).*(xlamS(3,:)-xnode(3))+...
    2*M(2,3)*(xlamS(2,:)-xnode(2)).*(xlamS(3,:)-xnode(3)));
x1=zeros(Nphi*Ntheta,nT);
x2=zeros(Nphi*Ntheta,nT);
x3=zeros(Nphi*Ntheta,nT);
x4=zeros(Nphi*Ntheta,nT);
x5=zeros(Nphi*Ntheta,nT);
x6=zeros(Nphi*Ntheta,nT);
x7=zeros(Nphi*Ntheta,nT);
pos=1:1:(Nphi*Ntheta);
delta=0;

for j=1:nT
    t0=(j-1)*h;
    xlamS2=rk4(t0,h,xlamS);

    I1=find((xlamS2(1,:)>xmax)|(xlamS2(1,:)<xmin)|(xlamS2(2,:)<ymin)|(xlamS2(2,:)>ymax)|(xlamS2(3,:)<zmin)|(xlamS2(3,:)>zmax));
    if isempty(I1)==0
        pos(I1)=[];
        xlamS2(:,I1)=[];
    end
    
    Np(pos)=Np(pos)+1;
    
    x1(pos,j)=xlamS2(1,:)';
    x2(pos,j)=xlamS2(2,:)';
    x3(pos,j)=xlamS2(3,:)';
    x4(pos,j)=xlamS2(4,:)';
    x5(pos,j)=xlamS2(5,:)';
    x6(pos,j)=xlamS2(6,:)';
    x7(pos,j)=xlamS2(7,:)';
    
    if isempty(pos)
        break;
    end
    xlamS=xlamS2;
end

figure;
for i=1:Nphi*Ntheta
    plot3(x1(i,1:Np(i)),x2(i,1:Np(i)),x3(i,1:Np(i)));
    hold on
end
hold off

xN1=[];
xN2=[];
xN3=[];
xN4=[];
xN5=[];
xN6=[];
xN7=[];
for i=1:Nphi*Ntheta
    xN1=[xN1 x1(i,1:Np(i))];
    xN2=[xN2 x2(i,1:Np(i))];
    xN3=[xN3 x3(i,1:Np(i))];
    xN4=[xN4 x4(i,1:Np(i))];
    xN5=[xN5 x5(i,1:Np(i))];
    xN6=[xN6 x6(i,1:Np(i))];
    xN7=[xN7 x7(i,1:Np(i))];
end

Nx=20;
Ny=20;
Nz=20;
xlin=linspace(xmin,xmax,Nx+1);
ylin=linspace(ymin,ymax,Ny+1);
zlin=linspace(zmin,zmax,Nz+1);
xdata1=[];
xdata2=[];
xdata3=[];
xdata4=[];
xdata5=[];
xdata6=[];
xdata7=[];
for i=1:Nx
    I1=(xN1>=xlin(i))&(xN1<xlin(i+1));
    for j=1:Ny
        I2=(xN2>=ylin(j))&(xN2<ylin(j+1));
        for k=1:Nz
            I3=(xN3>=zlin(k))&(xN3<zlin(k+1));
            I=I1&I2&I3;
            xt1=xN1(I);
            xt2=xN2(I);
            xt3=xN3(I);
            xt4=xN4(I);
            xt5=xN5(I);
            xt6=xN6(I);
            xt7=xN7(I);
            if ~isempty(xt7)
                [m,I0]=min(xt7);
                xdata1=[xdata1 xt1(I0)];
                xdata2=[xdata2 xt2(I0)];
                xdata3=[xdata3 xt3(I0)];
                xdata4=[xdata4 xt4(I0)];
                xdata5=[xdata5 xt5(I0)];
                xdata6=[xdata6 xt6(I0)];
                xdata7=[xdata7 xt7(I0)];
            end
        end
        
    end
end
Ndata=length(xdata1);

figure;
plot3(xdata1,xdata2,xdata3,'*');

path = sprintf('xdata1.mat');
save(path,'xdata1');
path = sprintf('xdata2.mat');
save(path,'xdata2');
path = sprintf('xdata3.mat');
save(path,'xdata3');
path = sprintf('xdata4.mat');
save(path,'xdata4');
path = sprintf('xdata5.mat');
save(path,'xdata5');
path = sprintf('xdata6.mat');
save(path,'xdata6');
path = sprintf('xdata7.mat');
save(path,'xdata7');
% path = sprintf('Ndata.mat');
% save(path,'Ndata');

% figure;
% plot3(xdata1,xdata2,xdata5);

N=100;
xlin=linspace(xmin,xmax,N);
ylin=linspace(ymin,ymax,N);
zlin=linspace(zmin,zmax,N);
[xmesh,ymesh,zmesh]=meshgrid(xlin,ylin,zlin);
Strue=xmesh.^4-2*xmesh.^2+ymesh.^2+zmesh.^2+1;
% figure;
% mesh(xmesh,ymesh,Strue);

xtest1=reshape(xmesh,1,N^3);
xtest2=reshape(ymesh,1,N^3);
xtest3=reshape(zmesh,1,N^3);
path = sprintf('xtest1.mat');
save(path,'xtest1');
path = sprintf('xtest2.mat');
save(path,'xtest2');
path = sprintf('xtest3.mat');
save(path,'xtest3');

% figure;
% plot(LOSS);
% 
% Slearn=reshape(Stest,N,N);
% figure;
% mesh(xmesh,ymesh,Slearn);

xva1=[-1.5,-1.5,-1.5,-1,-1,-1,0,0,0];
xva2=[-0.6,0,0.6,-0.6,0,0.6,-0.6,0,0.6];
xva3=[-0.6,0,0.6,-0.6,0,0.6,-0.6,0,0.6];
xva4=4*xva1.^3-4*xva1;
xva5=2*xva2;
xva6=2*xva3;
xva7=xva1.^4-2*xva1.^2+xva2.^2+xva3.^2+1;

% figure;
% plot(MPEP(:,1),MPEP(:,2));
