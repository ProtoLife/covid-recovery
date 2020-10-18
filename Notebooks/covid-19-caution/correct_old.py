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

2nd phase

 for t in range(7,n):                               # speed up by ignoring correction to first 7 pts with too little data
                    nt = 7  # min(7,t)
                    nft= 7. # float(nt)
                    ne = 5. #nft-2.                                # two points give no deviation
                    x = times[t-nt:t]                              # t-nt up to and including t-1
                    # y = data_cc[t-nt:t]                          # 
                    y = data_cc[t-nt:t]                             # rather than use data_cc we may use the corrected values to avoid glitch errors
                    ys = data_ccs2[t-nt:t]

                    # sl, y0, r, p, se = stats.linregress(x,y)       # regression fit to unsmoothed data
                    # sls, y0s, rs, ps, ses = stats.linregress(x,ys) # regression fit to smoothed data
                    # l = np.array(y0+x*sl)                          # unsmoothed regression line pts
                    # ls = np.array(y0s+x*sls)                       # smoothed regression line pts
                    # sigmar =  np.sqrt(np.sum(np.square(y-l)/ne))               
                    # sigmars =  np.sqrt(np.sum(np.square(ys-ls)/ne))
                    # yp = y0+times[t]*sl                            # predicted value at t from unsmoothed data
                    # yps = y0s+times[t]*sls                         # predicted value at t from smoothed data
                    # yps = max(0.,yps)
                    yp = np.mean(y)
                    sigmar = np.std(y)
                    yps = np.mean(ys)
                    sigmars = np.std(ys)
                    delta = data_cc[t]-yp
                    adelta = delta-np.sign(delta)*sigmar
                    week = week - cor_ts[t-7] + data_cc[t]         # rolling sum of last 7 : initially using data_cc for estimate, later corrected
                    # deltas = (week/7.-yps)                         # jump in smoothed curve (from predicted value)
                    deltas = data_ccs[t]-yps
                    adeltas = deltas - np.sign(deltas)*sigmars
                    adeltas7 = adeltas*7.                          # change to data_cc that would give this jump in smoothed rolling average
                    if sigmars > 0.1 and np.abs(delta) > 10.:
                        if np.abs(deltas) > 8.*sigmars: # and np.abs(delta) > 3.*sigmar:            # try correction
                                      # do correction : limit deviation to sigmar
                            file.write("%s,\"%s\",%d,%f,%f,%f\n" % (dtype,cc,t,yps,deltas,sigmars))
                            cor_ts[t] =  data_cc[t] - adeltas7
                            data_ccs2[t] = data_ccs[t] - deltas
                            if False:
                                tsum = np.sum(cor_ts[max(0,t-31):t-1]) 
                                if adeltas7 > 0:
                                    if tsum > 0:                   # redistribute over previous month proportional to counts                     
                                        inv_tsum = 1./tsum 
                                        for t1 in range(max(0,t-31),t):
                                            cor_ts[t1] = cor_ts[t1]*(1. + adeltas7 *inv_tsum)
                                    else:
                                        inv_tsum = 1./(t-max(0,t-31))# replace by linear ramp       
                                        for t1 in range(max(0,t-31),t):
                                            cor_ts[t1] = cor_ts[t1] + adeltas7 * inv_tsum
                                elif tsum > -delta:
                                        inv_tsum = 1./tsum 
                                        for t1 in range(max(0,t-31),t):
                                            cor_ts[t1] = cor_ts[t1]*(1. + adeltas7 *inv_tsum)                           
                                else:
                                   file.write("%s,\"%s\",%d,%f,%f,%f,no redistr\n" % (dtype,cc,t,yps,adeltas,sigmars))
                                   # print('redistribution not possible')
                                   cor_ts[t] = data_cc[t]
                        else:
                            cor_ts[t] = data_cc[t]
                    else:
                        cor_ts[t] = data_cc[t]
                    week = week + cor_ts[t] - data_cc[t]
                data_cor.update({cc:cor_ts})