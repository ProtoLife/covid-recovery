def clust(clustering_a,clustering_b,colors_a,colors_b,relabel=True,merge=True): 
    """ relables clustering b to match clustering a
        if more than one cluster in a optimally matches a particular cluster in b, then color of b is merger of colors in a
        if more than one cluster in b optimally matches a particular cluster in a, then colors in a merged and split for b
        inputs: clustering_a,b are lists of cluster labels by country, colors_a,b are lists of rgb colors by country in same order  
        returns: newcolors_b in rgb format
        NB. colors_b are only used to preserve s,v values relating to probs of cluster membership for b in final colors
        NB. the hues of b_cols are determined by the matching of clustering b with clustering a
        NB. all elts of same cluster have the same hue
    """
    labels_a = list(set(clustering_a))
    labels_b = list(set(clustering_b))
    newcolors_b = np.zeros((len(colors_b),3),dtype=float)
    newcolors_b[:,:] = colors_b[:,:]   # initialized as copy of colors_b, colors for each country in clustering b
            
    a_to_b = {}
    b_to_a = {}
    a_cols = {}
    b_cols = {}
        
    #    a_to_b mapping of labels a to the label b (+ its match score in a tuple) with largest matching score: ratio of intersecting countries to union
    #    maxvals_a_to_b are list of max scores for each label in labels_a
    #    reorder_a is the largest to smallest order of max scores 
    #    labels_a_sort is labels_a reordered by reorder_a :i.e. the labels in a with the best matches to a label in b first
    
    for a in labels_a:
        maxscore = 0
        maxlab = -2
        for b in labels_b:
            score = score_int_union(matchset(clustering_a,a),matchset(clustering_b,b)) # length intersection divided by length union (result 0. to 1. for identity)
            if score > maxscore:
                maxscore = score
                maxlab = b
        a_to_b.update({a:(maxlab,maxscore)})
    maxvals_a_to_b = [a_to_b[a][1] for a in labels_a]
    reorder_a = np.flip(np.argsort(maxvals_a_to_b))
    labels_a_sort = [labels_a[r] for r in list(reorder_a)]

    # same as above for b_to_a
    for b in labels_b:
        maxscore = 0
        maxlab = -2
        for a in labels_a:
            score = score_int_union(matchset(clustering_a,a),matchset(clustering_b,b))
            if score > maxscore:
                maxscore = score
                maxlab = a
        b_to_a.update({b:(maxlab,maxscore)})
    maxvals_b_to_a = [b_to_a[b][1] for b in labels_b]
    reorder_b = np.flip(np.argsort(maxvals_b_to_a))
    labels_b_sort = [labels_b[r] for r in list(reorder_b)]    

    #print('before relabel')

    # relabeling uses labels_b_sort, labels_a_sort, a_to_b, as well as colors_a,b and clustering_a,b
    if relabel:
        #print('Debug in relabel: labels_b_sort',labels_b_sort)
        for b in labels_b_sort:   # first adjust colors_b to match mapped clusters from a (transfer and merge)
            amap = [a for a in labels_a_sort if a_to_b[a][0] == b] # the labels a that prefer b as best match
            #print('Debug in relabel: b',b,'amap',amap)
            for a in amap:
                alist = matchset(clustering_a,a) # the positions in country list with label a (non empty since a is a label of clustering_a)
                # a_cols[a] = colors_a[alist[0]]   # dictionary of colors chosen as color of first country in alist of a
                a_cols.update({(b,a) : mpcolors.hsv_to_rgb(color_mean_rgb_to_hsv([colors_a[al] for al in alist]))})   # average color of alist for b chosen as color
                #for al in alist:
                #    print('              al,rgb,hsv',al,colors_a[al],mpcolors.rgb_to_hsv(colors_a[al]))
                #print('    Debug in relabel: b,a,a_cols[(b,a)]',b,a,a_cols[(b,a)],'->hsv',mpcolors.rgb_to_hsv(a_cols[(b,a)]))
            blist = matchset(clustering_b,b)     # the positions in country list with label b
            amap_t = list(set(amap)-set([-1]))   # the labels of real clusters (excluding unclustered set with label -1) that prefer b 
            #print('Debug in relabel: amap_t',amap_t)
            if len(amap_t) > 0: # some non-unclustered (ie not -1) clusters that prefer to map to b
                # h = sum([mpcolors.rgb_to_hsv(a_cols[a])[0] for a in amap])/len(amap) # average hue from amap
                h = color_mean_rgb_to_hsv([a_cols[(b,a)] for a in amap_t],[a_to_b[a][1] for a in amap_t])[0]
                for j in blist:                             # indices of countries with label b
                    s = mpcolors.rgb_to_hsv(colors_b[j])[1] # take s saturation from b
                    v = mpcolors.rgb_to_hsv(colors_b[j])[2] # take v value from b
                    newcolors_b[j,:] = mpcolors.hsv_to_rgb([h,s,v]) # back to rgb  
            # b_cols[b] = newcolors_b[blist[0]] # first matching elt colour (to extract hue)
            b_cols[b] = mpcolors.hsv_to_rgb(color_mean_rgb_to_hsv([newcolors_b[bl] for bl in blist]))   # average color of blist chosen as color

    #print('before merge')

            
    if merge:
        for a in labels_a_sort:   # now readjust colors in b that both map to same a (split)
            bmap = [b for b in labels_b_sort if b_to_a[b][0] == a]
            #print('Debug in merge: a',a,'bmap',bmap)
            if len(bmap)>1:                          
                for i,b in enumerate(bmap):
                    blist = matchset(clustering_b,b)
                    # h = (mpcolors.rgb_to_hsv(b_cols[b])[0] + mpcolors.rgb_to_hsv(a_cols[a])[0])/2
                    #print('    Debug in merge: a,b,a_cols[(b,a)]',a,b,a_cols[(b,a)],'->hsv',mpcolors.rgb_to_hsv(a_cols[(b,a)]))
                    if (b,a) in list(a_cols.keys()):
                        h,s0,v0 = color_mean_rgb_to_hsv([b_cols[b],a_cols[(b,a)]]) # mean hue
                        #print('    Debug in merge doing merge: a,b,b_cols[b],a_cols[(b,a)]',a,b,b_cols[b],a_cols[(b,a)])
                    else:
                        h,s0,v0 = mpcolors.rgb_to_hsv(b_cols[b])                      
                    for j in blist:                     
                        # s = mpcolors.rgb_to_hsv(b_cols[b])[1] # take s saturation from b   # these two lines cause all elts to have same value as first for s and v
                        # v = mpcolors.rgb_to_hsv(b_cols[b])[2] # take v from b
                        s = mpcolors.rgb_to_hsv(colors_b[j])[1] # take s saturation from b
                        v = mpcolors.rgb_to_hsv(colors_b[j])[2] # take v from b
                        newcolors_b[j,:]= mpcolors.hsv_to_rgb([h,s,v])
                    b_cols[b]=mpcolors.hsv_to_rgb([h,s0,v0])
    return newcolors_b