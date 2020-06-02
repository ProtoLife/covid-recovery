function dx=simpleGrowth(t,x,r,p,flag1)

dx=zeros(1,1);

if flag1==1 % exponential growth model
    p=1;
    dx(1,1)=r*x;
    
elseif flag1==2 % generalized-growth model
    
    dx(1,1)=r*x^p;
    
end





