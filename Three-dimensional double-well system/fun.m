function y=fun(~,x)

%%%% Computing the quasipotential for nongradient SDEs in 3D

global rou
x1=x(1,:);
x2=x(2,:);
x3=x(3,:);
x4=x(4,:);
x5=x(5,:);
x6=x(6,:);

y=zeros(size(x));
y(1,:)=-2*(x1.^3-x1)-rou*(x2+x3)+x4;
y(2,:)=-x2+2*rou*(x1.^3-x1)+x5;
y(3,:)=-x3+2*rou*(x1.^3-x1)+x6;
y(4,:)=2*(3*x1.^2-1).*x4-2*rou*(3*x1.^2-1).*x5-2*rou*(3*x1.^2-1).*x6;
y(5,:)=rou*x4+x5;
y(6,:)=rou*x5+x6;
y(7,:)=1/2*x4.^2+1/2*x5.^2+1/2*x6.^2;
