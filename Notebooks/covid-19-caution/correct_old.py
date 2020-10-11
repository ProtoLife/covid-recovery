        for cc in data_diff:
            if cc == 'dates':
                data_cor.update({'dates':data_diff['dates']})
            else:
                data_cc = data_diff[cc] 
                data_ccs = data_sm[cc] 
                cor_ts = np.zeros(n,dtype=float)
                week = 0.
                wvar = 0
                sum = 0
                wvars = 0
                cor_ts[0] = data_cc[0]
                for t in range(1,n): 
                    nt = min(7,t)
                    nft=float(nt)                 
                    wvar = wvar + data_cc[t-1]*data_cc[t-1]
                    wvars = wvars + data_ccs[t-1]*data_ccs[t-1]
                    sum = sum + data_ccs[t-1]
                    if t > 7:
                        sum = sum - data_ccs[t-8]
                        wvar = wvar - data_cc[t-8]*data_cc[t-8]
                        wvars = wvars - data_ccs[t-8]*data_ccs[t-8]
                    mean = data_ccs[t-1]
                    means = sum/nft
                    var = wvar/nft - mean*mean
                    vars = wvars/nft - means*means

                    if var >= 0.:
                        sigma = np.sqrt(var)
                    elif -var < 0.000001:
                        sigma = 0.
                    else:
                        print('invalid argument var to sqrt',var,'at',dtype,cc,'time',t,'nt mean val var',nt,mean,data_cc[t],var+mean*mean)
                        sigma = 0.

                    if vars >= 0.:
                        sigmas = np.sqrt(vars)
                    elif -vars < 0.000001:
                        sigmas = 0.
                    else:
                        print('invalid argument vars to sqrt',vars,'at',dtype,cc,'time',t,'nt mean val vars',nt,means,data_ccs[t],vars+means*means)
                        sigma = 0.

                    if sigma > 0.001 and sigmas > 0.001:
                        delta = data_cc[t]-mean
                        deltas = data_ccs[t] - means
                        if np.abs(delta) < 3.*sigma or np.abs(deltas) < 2.*sigmas: # no correction
                            cor_ts[t] = data_cc[t]
                        else:                   # do correction
                            if delta > 0.:
                                sign = 1
                            else:
                                sign = -1
                            corr = delta-sign*sigma
                            file.write("%s,\"%s\",%d,%f,%f,%f\n" % (dtype,cc,t,mean,corr,sigma))
                            # print('Correction',dtype,cc,'time',t,'mean',mean,'correction',corr,'sigma',sigma)
                            cor_ts[t] =  data_cc[t] - corr
                            tsum = np.sum(cor_ts[max(0,t-31):t-1])   # redistribute over previous month (could also choose 6-8 weeks)
                            if tsum > 0:
                                for t1 in range(max(0,t-31),t):
                                    cor_ts[t1] = cor_ts[t1]*(1. + corr/tsum)
                    else:
                        cor_ts[t] = data_cc[t]
                data_cor.update({cc:cor_ts})        
        new_covid_ts.update({new_dtype_corrected:data_cor})