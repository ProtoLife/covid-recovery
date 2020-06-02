function weights=sesweights(n,alpha1)

weights=ones(n,1);

if isnan(alpha1)==0 % weighted with simple exponential smoothing
    
    weights(1,1)=alpha1;
    
    for i=2:n
        weights(i,1)=weights(i-1)*(1-alpha1);
    end
    
    weights=weights(end:-1:1);
    
end

