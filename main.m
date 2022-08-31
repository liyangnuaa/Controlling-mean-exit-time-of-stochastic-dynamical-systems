clear;
clc;

xmin=-1.5;
xmax=0;
ymin=-0.6;
ymax=0.6;

global miu
miu=1;

xnode1=[-1;0];
xnode2=[1;0];
xsad=[0;0];
xnode=xnode1;

Nphi=2000;            % »·ÉÏ»®·Ö¾«¶È
phi=linspace(0,2*pi,Nphi);

A=[1-3*xnode(1)^2-miu*xnode(2)^2, -2*miu*xnode(1)*xnode(2);-2*xnode(1)*xnode(2), -(1+xnode(1)^2)];
B=zeros(4,4);
B(1:2,1:2)=A;
B(1:2,3:4)=[1 0;0 1];
B(3:4,3:4)=-A';
[Bv,Be]=eig(B);
Bv1=Bv(1:2,3:4);
Bv2=Bv(3:4,3:4);
M=real(Bv2/Bv1);

R=1e-2;
Nmap=1;
tf=20;
h=0.005;
nT=tf/h;
Np=zeros(1,Nphi);
xlamS=zeros(5,Nphi);
xlamS(1:2,:)=[xnode(1)+R*cos(phi);xnode(2)+R*sin(phi)];
xlamS(3:4,:)=M*[R*cos(phi);R*sin(phi)];
xlamS(5,:)=1/2*(M(1,1)*(xlamS(1,:)-xnode(1)).^2+M(2,2)*(xlamS(2,:)-xnode(2)).^2+2*M(1,2)*(xlamS(1,:)-xnode(1)).*(xlamS(2,:)-xnode(2)));
x1=zeros(Nphi,nT);
x2=zeros(Nphi,nT);
x3=zeros(Nphi,nT);
x4=zeros(Nphi,nT);
x5=zeros(Nphi,nT);
% xyS=zeros(3,Nphi);
pos=1:1:Nphi;
delta=0;

for j=1:nT
    t0=(j-1)*h;
    xlamS2=rk4(t0,h,xlamS);

    I1=find((xlamS2(1,:)>xmax)|(xlamS2(1,:)<xmin)|(xlamS2(2,:)<ymin)|(xlamS2(2,:)>ymax));
    if isempty(I1)==0
%         xyS(1,pos(I1))=xlamS2(1,I1);
%         xyS(2,pos(I1))=xlamS2(2,I1);
%         xyS(3,pos(I1))=xlamS2(5,I1);
        pos(I1)=[];
        xlamS2(:,I1)=[];
    end
    
    Np(pos)=Np(pos)+1;
    
    x1(pos,j)=xlamS2(1,:)';
    x2(pos,j)=xlamS2(2,:)';
    x3(pos,j)=xlamS2(3,:)';
    x4(pos,j)=xlamS2(4,:)';
    x5(pos,j)=xlamS2(5,:)';
    
    if isempty(pos)
        break;
    end
    xlamS=xlamS2;
end

% figure;
% for i=1:Nphi
%     plot(x1(i,1:Np(i)),x2(i,1:Np(i)),'m-');
%     hold on
% end
% hold off

xN1=[];
xN2=[];
xN3=[];
xN4=[];
xN5=[];
for i=1:Nphi
    xN1=[xN1 x1(i,1:Np(i))];
    xN2=[xN2 x2(i,1:Np(i))];
    xN3=[xN3 x3(i,1:Np(i))];
    xN4=[xN4 x4(i,1:Np(i))];
    xN5=[xN5 x5(i,1:Np(i))];
end

Nx=20;
Ny=20;
xlin=linspace(xmin,xmax,Nx+1);
ylin=linspace(ymin,ymax,Ny+1);
xdata1=[];
xdata2=[];
xdata3=[];
xdata4=[];
xdata5=[];
for i=1:Nx
    I1=(xN1>=xlin(i))&(xN1<xlin(i+1));
    for j=1:Ny
        I2=(xN2>=ylin(j))&(xN2<ylin(j+1));
        I=I1&I2;
        xt1=xN1(I);
        xt2=xN2(I);
        xt3=xN3(I);
        xt4=xN4(I);
        xt5=xN5(I);
        if ~isempty(xt5)
            [m,I0]=min(xt5);
            xdata1=[xdata1 xt1(I0)];
            xdata2=[xdata2 xt2(I0)];
            xdata3=[xdata3 xt3(I0)];
            xdata4=[xdata4 xt4(I0)];
            xdata5=[xdata5 xt5(I0)];
        end
    end
end
Ndata=length(xdata1);

figure;
plot(xdata1,xdata2,'*');

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
% path = sprintf('Ndata.mat');
% save(path,'Ndata');

figure;
plot3(xdata1,xdata2,xdata5);

N=100;
xlin=linspace(xmin,xmax,N);
ylin=linspace(ymin,ymax,N);
[xmesh,ymesh]=meshgrid(xlin,ylin);
Strue=0.5*(xmesh.^2-1).^2+ymesh.^2.*(xmesh.^2+1);
% figure;
% mesh(xmesh,ymesh,Strue);

xtest1=reshape(xmesh,1,N^2);
xtest2=reshape(ymesh,1,N^2);
path = sprintf('xtest1.mat');
save(path,'xtest1');
path = sprintf('xtest2.mat');
save(path,'xtest2');

figure;
plot(LOSS);

Slearn=reshape(Stest(:,3),N,N);
figure;
mesh(xmesh,ymesh,Slearn);

xva1=[-1.5,-1.5,-1.5,-1,-1,-1,0,0,0];
xva2=[-0.8,0,0.8,-0.8,0,0.8,-0.8,0,0.8];
xva3=2*xva1.*(xva1.^2-1)+2*xva1.*xva2.^2;
xva4=2*(xva1.^2+1).*xva2;
xva5=0.5*(xva1.^2-1).^2+xva2.^2.*(xva1.^2+1);

% figure;
% plot(MPEP(:,1),MPEP(:,2));
