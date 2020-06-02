function y=plotSimpleGrowth1(z,timevect,I0,flag1,weights)

r=z(1);
p=z(2);

[r p]

IC=I0; % initial condition (number of cases)

[t,x]=ode45(@simpleGrowth,timevect,IC,[],r,p,flag1);

incidence1=[x(1,1);diff(x(:,1))];

y=weights.*incidence1;

