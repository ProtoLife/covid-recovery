#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib import colors as mpcolors

import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd

# Jupyter Specifics
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display, HTML
from ipywidgets.widgets import interact, interactive, IntSlider, FloatSlider, Layout, ToggleButton, ToggleButtons, fixed, Output
display(HTML("<style>.container { width:100% !important; }</style>"))
style = {'description_width': '100px'}
slider_layout = Layout(width='99%')

import skfda
from skfda.datasets import fetch_growth
from skfda.exploratory.visualization import plot_fpca_perturbation_graphs
from skfda.preprocessing.dim_reduction.projection import FPCA
from skfda.representation.basis import BSpline, Fourier, Monomial


import hdbscan
import warnings
import math

from tqdm.notebook import tqdm  # progress bars

## map stuff
import ipyleaflet
import json
import geopandas as gpd
import pickle as pk
import os
import requests
from ipywidgets import link, FloatSlider, HTML
from branca.colormap import linear
from matplotlib import colors as mpcolors

def corcl(a,b):
    if len(set(a)) > 0 or len(set(b)) > 0:
        return len(set(a).intersection(set(b)))/float(len(set(a).union(set(b))))
    else:
        return 1 

def match1(a,x):
    rtn = [1 if a[i] == x else 0 for i in range(len(a)) ]
    return rtn

def rescale(v,d):
    """ functional form of correction factor using simple inversion formula
        for with v2'=1/(1-v2) the dimensionality correction v = v2 * v2'/(v2'+d/2-1)
        projecting equivalent validity at dim = 2"""
    if d > 12.:
        d = 12.
    logd = np.log(0.5*d)
    return v*(1.+logd)/(1.+v*logd)

def score_int(a,b):
    if len(set(a)) > 0 or len(set(b)) > 0:
        return len(set(a).intersection(set(b)))
    else:
        return 0 
    
def score_int_union(a,b):
    if len(set(a)) > 0 or len(set(b)) > 0:
        return len(set(a)&set(b))/len(set(a)|set(b))  # length intersection divided by length union
    else:
        return 0 
    
def matchset(a,x):
    rtn = [i for i in range(len(a)) if a[i] == x]
    return rtn

def closest_hue(hue,huelist):
    mindist = 2.
    imin = -1
    for i,h in enumerate(huelist):
        if h > hue:
            dist = min(h-hue,hue+1-h)
        else:
            dist = min(hue-h,h+1-hue)
        if dist < mindist:
            mindist = dist
            imin = i
    return imin

def color_mean_rgb_to_hsv(rgb_colours,weights=None,modal=False): 
    """ the hue is a circular quantity, so mean needs care
        see https://en.wikipedia.org/wiki/Mean_of_circular_quantities
        inputs: rgb_colors 1D array of rgb colours with entries [r,g,b]
                weights: None,'all' or same length array of weights in 0. to 1. for biasing entries
                modal: if Ture then chose hue as mode of hues, otherwise circular mean
    """
    pi = np.pi
    eps = 0.0001
    hsum = 0.
    ssum = 0.
    vsum = 0.
    asum = 0.
    bsum = 0.
    wsum = 0.
    hwsum = 0.
    
    if len(rgb_colours) == 0:
        print('Error in color_mean_rgb_to_hsv: empty list of rgb_colours')
        return [0.,0.,0.]

    if weights == None:
        weights = [1. if mpcolors.rgb_to_hsv(c)[1] > 0 else 0. for c in rgb_colours] # designed to exclude -1 unclustered colours
        if np.sum(np.array(weights)) < eps:
            weights = [1. for c in rgb_colours]
    elif weights == 'all':
        weights = [1. for c in rgb_colours]
    
    hdic = {}
    for i,c in enumerate(rgb_colours):
        hsvcol = mpcolors.rgb_to_hsv(c)
        h = hsvcol[0]
        s = hsvcol[1]
        v = hsvcol[2]
        if s > eps and v > eps:
            asum = asum + np.sin(h*2.*pi)*weights[i]
            bsum = bsum + np.cos(h*2.*pi)*weights[i]
            hwsum = hwsum + weights[i]
            if h in hdic:
                hdic.update({h:hdic[h]+1})
            else:
                hdic.update({h:1})
        ssum = ssum + hsvcol[1]*weights[i]
        vsum = vsum + hsvcol[2]*weights[i]
        wsum = wsum + weights[i]
    if modal:
        hvals = list(hdic.keys())    
        hcnts = [hdic[h1] for h1 in hvals]
        if len(hcnts) > 0:
            hmaxcnt = np.argmax(np.array(hcnts)) # problem if hcnts is empty sequence
        else:
            hmaxcnt = None
    if modal and len(hcnts)>0 and hcnts[hmaxcnt] >= len(rgb_colours)/4:
        h = hvals[hmaxcnt]
        # print('using modal hue %f with cnts %d',h,hcnts[hmaxcnt])    
    elif hwsum > eps:
        asum = asum/hwsum
        bsum = bsum/hwsum
        h = np.arctan2(asum,bsum)/(2.*pi)
        if h < 0.:
            h = 1.+h
    else:
        h = 0.
    if wsum > eps:
        s = ssum/wsum
        v = vsum/wsum
    else:
        print('Error in color_mean_rgb_to_hsv: 0 wsum')
        s = 0.
        v = 0.
    # print(rgb_colours,'mean',mpcolors.hsv_to_rgb([h,s,v]))
    if h < 0.:
        print('error in color_mean, hue out of range',h)
        h = 0.
    if h > 1.:
        print('error in color_mean, hue out of range',h)
        h = 1.
    return [h,s,v]
        
def size_order(clusterings):
    """ relabel clusters in each clustering in order of increasing size"""
    clusterings_o = np.zeros(clusterings.shape,dtype = int) 
    for i,clustering in enumerate(clusterings):
        labels = list(set(clustering)-set([-1]))
        sizes = np.zeros(len(labels),dtype = int)
        for j,lab in enumerate(labels):
            sizes[j] = len(matchset(clustering,lab))
        order = np.flip(np.argsort(sizes))
        clusterings_o[i,:] = [order[c] if c != -1 else c for c in clustering]
    return clusterings_o

def clust_assign(clustering_a,clustering_b,colors_a,colors_b):
    """ relables clustering b to match clustering a optimally
        according tot he Hungarian algorithm, implemented in scipy
    """    
    labels_a = list(set(clustering_a))
    labels_b = list(set(clustering_b))
    scores = np.zeros((len(labels_a),len(labels_b)),dtype=float)
    for i,a in enumerate(labels_a):
        for j,b in enumerate(labels_b):
            scores[i,j] = score_int_union(matchset(clustering_a,a),matchset(clustering_b,b)) # length intersection divided by length union (result 0. to 1. for identity)
    assign_a_to_b,assign_b_to_a=scipy.optimize.linear_sum_assignment(scores)
    dic_a_2_b = {labels_a[i]:labels_b[j] for i,j in enumerate(assign_a_to_b)}
    return dic_a_2_b


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
        for b in labels_b_sort:   # first adjust colors_b to match mapped clusters from a (transfer and merge)
            amap = [a for a in labels_a_sort if a_to_b[a][0] == b] # the labels a that prefer b as best match
            for a in amap:
                alist = matchset(clustering_a,a) # the positions in country list with label a (non empty since a is a label of clustering_a)
                a_cols.update({(b,a) : mpcolors.hsv_to_rgb(color_mean_rgb_to_hsv([colors_a[al] for al in alist]))})   # average color of alist for b chosen as color
                # print('in relabel a,b,a_cols',a,b,a_cols[(b,a)])
            blist = matchset(clustering_b,b)     # the positions in country list with label b
            amap_t = list(set(amap)-set([-1]))   # the labels of real clusters (excluding unclustered set with label -1) that prefer b 
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
            if len(bmap)>1:                          
                for i,b in enumerate(bmap):
                    blist = matchset(clustering_b,b)
                    # h = (mpcolors.rgb_to_hsv(b_cols[b])[0] + mpcolors.rgb_to_hsv(a_cols[a])[0])/2
                    if (b,a) in list(a_cols.keys()):
                        h,s0,v0 = color_mean_rgb_to_hsv([b_cols[b],a_cols[(b,a)]]) # mean of current color and that of a class that prefers this b
                    else:
                        # h = mpcolors.rgb_to_hsv(colors_b[j])[0]
                        h,s0,v0 = mpcolors.rgb_to_hsv(b_cols[b])
                    for j in blist:                     
                        # s = mpcolors.rgb_to_hsv(b_cols[b])[1] # take s saturation from b   # these two lines cause all elts to have same value as first for s and v
                        # v = mpcolors.rgb_to_hsv(b_cols[b])[2] # take v from b
                        s = mpcolors.rgb_to_hsv(colors_b[j])[1] # take s saturation from b
                        v = mpcolors.rgb_to_hsv(colors_b[j])[2] # take v from b
                        newcolors_b[j,:]= mpcolors.hsv_to_rgb([h,s,v])
                    b_cols[b]=mpcolors.hsv_to_rgb([h,s0,v0])

    return newcolors_b

def clust_lsa(clustering_a,clustering_b,colors_a,colors_b,base_colors=None,relabel=True,merge=True): 
    """ relables clustering b to match clustering a, optimally using first linear_sum_assignment, 
        then augmenting with new clusters for poorly aligned clusters
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        inputs: clustering_a,b are lists of cluster labels by country, colors_a,b are lists of rgb colors by country in same order  
        returns: newcolors_b in rgb format
        NB. colors_b are only used to preserve s,v values relating to probs of cluster membership for b in final colors
        NB. the hues of b_cols are determined by the matching of clustering b with clustering a
        NB. all elts of same cluster have the same hue
    """
    # print('in clust_lsa')
    labels_a = list(set(clustering_a))
    labels_b = list(set(clustering_b))
    labels_a_clus = list(set(clustering_a)-set([-1])) # all except unclustered class
    labels_b_clus = list(set(clustering_b)-set([-1]))
    
    # do linear_sum_assignment based on cost matrix: 1-score_int_union
    scores = np.zeros((len(labels_a_clus),len(labels_b_clus)),dtype=float)
    score_dict = {}
    for i,a in enumerate(labels_a_clus):
        for j,b in enumerate(labels_b_clus): # length intersection divided by length union (result 0. to 1. for identity)
            scores[i,j] = 1-score_int_union(matchset(clustering_a,a),matchset(clustering_b,b)) 
            score_dict[(a,b)] = scores[i,j]
    row_ind,col_ind=linear_sum_assignment(scores)
    
    # construct forward and backward dictionary assignments
    if len(row_ind) != len(col_ind):
        print('Error: row and col indices have different lengths',row_ind,col_ind)
    dic_a_2_b = {labels_a_clus[row_ind[i]]:labels_b_clus[col_ind[i]] for i in range(len(row_ind))}
    dic_a_2_b.update({-1:-1})
    dic_b_2_a = {labels_b_clus[col_ind[i]]:labels_a_clus[row_ind[i]] for i in range(len(row_ind))}
    dic_b_2_a.update({-1:-1})
    
    # introduce new labels for poor matching labels in complete assignment
    maxlabel = max(labels_a) 
    relabel_b = {} 
    for a in labels_a:
        if a not in dic_a_2_b.keys():
            dic_a_2_b.update({a:None})
        else:
            relabel_b[dic_a_2_b[a]]=a   
    # print('dic a_2_b',dic_a_2_b)
    # print('dic b_2_a',dic_b_2_a)
    # print('relabel_b I',relabel_b)
    for b in labels_b: #unmatched labels b are given new cluster labels
        if b not in relabel_b.keys():
            maxlabel = maxlabel+1
            relabel_b[b]=maxlabel
        elif b != -1:
            if score_dict[(dic_b_2_a[b] ,b)] > 0.8:  # insufficient match, new cluster name
                # print('new label',dic_b_2_a[b],b,scoredict[(dic_b_2_a[b] ,b)])
                maxlabel = maxlabel+1
                relabel_b[b]=maxlabel
    # print('relabel_b II',relabel_b)
    new_labels_b = np.array([relabel_b[b] for b in labels_b])
    
    newcolors_b = np.zeros((len(colors_b),3),dtype=float)
    newcolors_b[:,:] = colors_b[:,:]   # initialized as copy of colors_b, colors for each country in clustering b
            

    # relabel colours
    if relabel:    
        for b in labels_b:   # first adjust colors_b to match mapped clusters from a (transfer and merge)
            if relabel_b[b] in labels_a:
                a = dic_b_2_a[b]
                alist = matchset(clustering_a,a)
                newcol =  colors_a[alist[0]] 
            else:
                newcol = base_colors[1+relabel_b[b]]
                # print('new color for b from',b,'to',relabel_b[b],'entry +1',newcol)
            h =  mpcolors.rgb_to_hsv(newcol)[0]
            for j in matchset(clustering_b,b):      # indices of countries with label b
                s = mpcolors.rgb_to_hsv(colors_b[j])[1] # take s saturation from b
                v = mpcolors.rgb_to_hsv(colors_b[j])[2] # take v value from b
                newcolors_b[j,:] = mpcolors.hsv_to_rgb([h,s,v]) # back to rgb  

    # no merge yet

    return newcolors_b

def cluster_map_colors(cons1,cons2,relabel=True,merge=True):
    """ recalculate colors of countries in consensus clustering cons2, based on alignment with clustering cons1
        input: two consensus clusterings with completed scans
               relabel abnd merge options (default True) as for clust 
        output: colors2 : the matched coloring of cons2
        side_effect : places inverse mapping iidx in cons2 to allow country order alignment
    """
    refc1 = cons1.refclustering
    refc2 = cons2.refclustering
    clusdat1 = np.array([cons1.clusdata[refc1][i] for i in cons1.sidx])
    clusdat2 =  np.array([cons2.clusdata[refc2][i] for i in cons1.sidx])   # NB not cons2.sidx
    if len(clusdat1) != len(clusdat2): 
        print('Error: country list lengths not equal')
        return None
    else: ncountries = len(clusdat2)
    cons2.iidx = [None]*ncountries
    for i, j in zip(range(ncountries), cons2.sidx): cons2.iidx[j] = i # undo cons2.idx reordering
    colors1 = np.array([cons1.basecolors[clus+1] for clus in clusdat1])
    colors2_c = np.array([cons2.basecolors[clusdat2[cons2.iidx[i]]+1] for i in range(ncountries)] ) 
    colors2_0 = np.array([colors2_c[cons2.sidx[i]] for i in range(ncountries)] )
    #colors1 = np.array(cons1.rgblist) # already ordered like scountries
    #colors2_c = np.array([cons2.rgblist[cons2.iidx[i]] for i in range(ncountries)] ) # change order back to match countries  # DEBUG 
    #colors2 = np.array([colors2_c[cons2.sidx[i]] for i in range(ncountries)] ) # change order to match scountries of cons1   # DEBUG
    # print(np.array(list(zip(clusdat2,mpcolors.rgb_to_hsv(colors2)))))
    colors2 = clust_lsa(clusdat1,clusdat2,colors1,colors2_0,base_colors=cons2.basecolors,relabel=relabel,merge=merge)
    #for i in range(len(colors2)):
    #    print(i,clusdat1[i],clusdat2[i],mpcolors.rgb_to_hsv(colors1[i]),mpcolors.rgb_to_hsv(colors2[i]))
    return colors1,colors2

def cmap_sankey(clus1,clus2,colors1,colors2,hue_only=True):
    cmap12 = {}
    for ci in set(clus1):
        for i,lab in enumerate(clus1): 
            if lab == ci:
                tmp = colors1[i]
                break
        cmap12.update({'a'+str(ci):tmp})

    for ci in set(clus2):
        for i,lab in enumerate(clus2): 
            if lab == ci:
                tmp = colors2[i]
                # print(ci,mpcolors.rgb_to_hsv(tmp))
                break  
        cmap12.update({'b'+str(ci):tmp})
    if hue_only:
        # reset colors to full saturation and value unless sat is 0
        cmap12h = {elt:mpcolors.hsv_to_rgb([mpcolors.rgb_to_hsv(cmap12[elt])[0],1,1] if mpcolors.rgb_to_hsv(cmap12[elt])[1]!=0 else [0,0,0]) for elt in cmap12}
    else:
        cmap12h = cmap12
    return cmap12h

def sankey(cons1,cons2,cons1_name='cons1',cons2_name='cons2',relabel=True,merge=True,hue_only=True):
    # extract refclustering data and order it according to the scountries list of cons1=cons 
    # set up dictionary lists of countries for each label
    if len(cons1.countries) != len(cons2.countries):
        print('Error: lengths of countries not equal',len(cons1.countries),len(cons2.countries))
        return
    clus1 = [cons1.clusdata[cons1.refclustering][i] for i in cons1.sidx]  # ordered like scountries
    clus2 = [cons2.clusdata[cons2.refclustering][i] for i in cons1.sidx]
    colors1,colors2=cluster_map_colors(cons1,cons2,relabel=relabel,merge=merge)
    cmap12=cmap_sankey(clus1,clus2,colors1,colors2,hue_only=hue_only)
    dic1 = {lab:[cc for i,cc in enumerate(cons1.scountries) if clus1[i]==lab] for lab in set(clus1)}
    dic2 = {lab:[cc for i,cc in enumerate(cons1.scountries) if clus2[i]==lab] for lab in set(clus2)}
    df = dic_compare(dic1,dic2)
    h1 = hv.Sankey(df,kdims=['c1','c2'],vdims=['val'])
    h1.opts(title=cons1_name+' vs '+cons2_name, cmap=cmap12,  node_color='index', edge_color='c1', node_alpha=1.0, edge_alpha=0.7)
    return h1

# the final cluster alignment
def plot_clusalign(countries,data,report,cols=None):
    fig,ax = plt.subplots(1,1,figsize=(10,24))
    if cols is not None:
        todel = list(set(range(data.shape[1])) - set(cols))
        data1 = np.delete(data,todel,1)
    else:
        data1 = data
    img = ax.imshow(data1)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries)
    if cols is None:
        rep = report
    else:
        rep = [report[i] for i in cols]
    ax.set_xticks(range(len(rep)))
    plt.setp(ax.get_xticklabels(), rotation='vertical', family='monospace')
    ax.set_xticklabels(rep,rotation='vertical')
    #plt.show()
    return fig


# Note that the colours are best understood as hue with value v = intensity related to membership prob
# note that unclustered points had probdata values of 0 formerly, now corrected to give outlier_score_
#
# We should be assigning countries to at least 4 categories : probably five.  Cluster 0,1,2 other cluster and no cluster (-1)
# Currently the code relies on the color assignments cluster 0 [1,0,0]  1 [0,1,0]  2 [0,0,1] and only works for 3 clusters.
# The unclustered color of [1,1,1] did not disrupt if the probability was always 0 : this will not work with outlier extension
# Other clusters higher in number were assigned rather biassedly to one of 0,1,2 : this needs fixing
# 

# count +1 for any RGB component
def cscore(crow,cols):
    rgbsc = [0.0]*3
    for j in cols:
        if crow[j][0] >0:
            rgbsc[0] = rgbsc[0]+1
        if crow[j][1] >0:
            rgbsc[1] = rgbsc[1]+1
        if crow[j][2] >0:
            rgbsc[2] = rgbsc[2]+1
    return rgbsc

# sum RGB components
def cscore_org(crow,cols):
    rgbsc = [0.0]*3
    for j in cols:
        rgbsc[0] = rgbsc[0]+crow[j][0]
        rgbsc[1] = rgbsc[1]+crow[j][1]
        rgbsc[2] = rgbsc[2]+crow[j][2]   
    return rgbsc

#sum weighted hues
def hscore_org(crow,cols):
    hsvmean = color_mean_rgb_to_hsv([crow[j] for j in cols],'all')
    return hsvmean

def hscore_mode_org(crow,cols):
    hsvmean = color_mean_rgb_to_hsv([crow[j] for j in cols],'all',modal=True)
    return hsvmean


def swizzle_old(countries,data,cols):
    rgb = [None]*len(countries)
    for i in range(len(countries)):
        for j in range(data.shape[1]):
            rgbsc = cscore(data[i,:,:],cols)
        rgb[i] = np.argmax(rgbsc)
    rtn = [None]*len(countries)
    cnt = 0
    print('-------blue---------')
    for i in range(len(rgb)):
        if rgb[i] == 2:  #blue
            rtn[cnt] = i
            print(cnt,i,countries[i])
            cnt = cnt+1
    print('-------green---------')

    for i in range(len(rgb)):
        if rgb[i] == 1:  # green
            rtn[cnt] = i
            print(cnt,i,countries[i])
            cnt = cnt+1    
    print('-------red---------')
    for i in range(len(rgb)):
        if rgb[i] == 0:  # red    
            rtn[cnt] = i
            print(cnt,i,countries[i])
            cnt = cnt+1
    print('cnt =',cnt)
    return rtn

def swizzleRGB(countries,data,cols):
    rgb = [None]*len(countries)
    for i in range(len(countries)):
        for j in range(data.shape[1]):
            rgbsc = cscore(data[i,:,:],cols)
        rgb[i] = np.argmax(rgbsc)
    rtn = {}
    rtn['R']=[]
    rtn['G']=[]
    rtn['B']=[]
    cnt = 0
    for i in range(len(rgb)):
        if rgb[i] == 2:  #blue
            rtn['B'].append(countries[i])
            cnt = cnt+1
    for i in range(len(rgb)):
        if rgb[i] == 1:  # green
            rtn['G'].append(countries[i])

            cnt = cnt+1    
    for i in range(len(rgb)):
        if rgb[i] == 0:  # red    
            rtn['R'].append(countries[i])
            cnt = cnt+1
    print('cnt =',cnt)
    return rtn


def swizzle2(countries,data,cols,refcol):
    eps = 0.0001
    clus = [None]*len(countries)
    rgblist = [None]*len(countries)
    hsvdic = {}
    hsvrefs = [mpcolors.rgb_to_hsv(c) for c in data[:,refcol]]
    huesref  = np.sort(list(set([hsv[0] for hsv in hsvrefs if hsv[1] > eps])))
    # print('huesref',huesref)
    for i in range(len(countries)):
        hsvsc = hscore_org(data[i,:,:],cols)
        hue = hsvsc[0]
        sat = hsvsc[1]
        if sat <= 0.5:  # mean is classed as unclustered
            clus[i] = -1
        else:
            clus[i] = closest_hue(hue,huesref)
        hsvdic.update({countries[i]:hsvsc})
        rgblist[i] = mpcolors.hsv_to_rgb(hsvsc)  
    # print('clus',clus,'len',len(clus))
    rtn = [None]*len(countries)
    cnt = 0
    for j in set(clus):
        print('-------class',j,'---------')
        for i in range(len(countries)):
            if clus[i] == j:  
                rtn[cnt] = i
                # print(cnt,i,countries[i],rgblist[i],hsvlist[i])
                print(cnt,i,countries[i])
                cnt = cnt+1
    print('cnt =',cnt)
    return rtn,rgblist,hsvdic


def swizzle3(countries,data,cols,refcol,basecolors,refdata,satthresh = 0.7):
    eps = 0.0001
    clus = [None]*len(countries)
    rgblist = [None]*len(countries)
    hsvdic = {}
    #hsvrefs = [mpcolors.rgb_to_hsv(c) for c in data[:,refcol]]
    refclus = np.sort(list(set(refdata))) # cluster classes in reference column
    #print('refclus',refclus)
    #huesref  = np.sort(list(set([hsv[0] for hsv in hsvrefs if hsv[1] > eps])))
    huesref = [mpcolors.rgb_to_hsv(basecolors[1+i])[0] for i in refclus if i != -1]
    #print('data shape',np.shape(data))
    #print('huesref',huesref)
    for i in range(len(countries)):
        # hsvsc = hscore_org(data[i,:,:],cols)
        # hsvsc = hscore_mode_org(data[i,:,:],cols) # using modal hue
        hsvsc = hscore_mode_org(data[i,:,:],cols)      # using color circle mean hue
        hue = hsvsc[0]
        sat = hsvsc[1]
        if sat <= satthresh:  # mean is classed as unclustered
            clus[i] = -1
        else:
            clus[i] = closest_hue(hue,huesref)
        #print(i,countries[i],hue,clus[i])
        hsvdic.update({countries[i]:hsvsc})
        rgblist[i] = mpcolors.hsv_to_rgb(hsvsc)  
    # print('clus',clus,'len',len(clus))
    rtn = [None]*len(countries)
    classes = [None]*len(countries)
    cnt = 0
    dic={}
    for j in set(clus):
        dic[j]=[]
        #print('-------class',j,'---------')
        for i in range(len(countries)):
            if clus[i] == j:
                classes[cnt] = j
                rtn[cnt] = i
                dic[j].append(countries[i])
                hsvdic[countries[i]] = [j]+hsvdic[countries[i]] # add class to hsvdic
                # print(cnt,i,countries[i],rgblist[i],hsvlist[i])
                #print(cnt,i,countries[i])
                cnt = cnt+1
    #print('cnt =',cnt)

    clus_argsort = np.lexsort((countries,clus))  # lexicographical sort of countries by reference clustering and name
    # swcountries = [countries[clus_argsort[i]] for i in range(len(countries))]  #  sorted country names as above
    swclasses = [clus[clus_argsort[i]] for i in range(len(countries))]  #  sorted country names as above
    swrgblist = [rgblist[clus_argsort[i]] for i in range(len(countries))]  #  sorted country names as above
    # return dic,swclasses,rtn,rgblist,hsvdic
    return dic,swclasses,clus_argsort,rgblist,hsvdic

def swizzle_class(countries,data,cols,refcol):
    clus = [None]*len(countries)
    huesref  = np.sort(list(set([mpcolors.rgb_to_hsv(c)[0] for c in data[:,refcol]])))
    # print('huesref',huesref)
    for i in range(len(countries)):
        hsvsc = hscore_org(data[i,:,:],cols)
        hue = hsvsc[0]
        sat = hsvsc[1]
        if sat <= 0.5:  # mean is classed as unclustered
            clus[i] = -1
        else:
            clus[i] = closest_hue(hue,huesref)
    rtn = {}
    for cl in set(clus):
        rtn[cl]=[]
    cnt = 0
    for j in set(clus):
        # print('-------class',j,'---------')
        for i in range(len(countries)):
            if clus[i] == j:
                rtn[j].append(countries[i])
                # print(cnt,i,countries[i])
                cnt = cnt+1
    print('cnt =',cnt)
    return rtn

def swizzleHSV(countries,data,cols,refcol):
    rtn = {}
    clus = [None]*len(countries)
    huesref  = np.sort(list(set([mpcolors.rgb_to_hsv(c)[0] for c in data[:,refcol]])))
    # print('huesref',huesref)
    for i in range(len(countries)):
        hsvsc = hscore_org(data[i,:,:],cols)
        hue = hsvsc[0]
        sat = hsvsc[1]
        if sat <= 0.5:  # mean is classed as unclustered
            clus[i] = -1
        else:
            clus[i] = closest_hue(hue,huesref)
        rtn[countries[i]]=(clus[i],hsvsc[0],hsvsc[1],hsvsc[2])
    return rtn

def dic_compare(dic1,dic2):
    df = pd.DataFrame(columns=['c1','c2','val'])
    cnt=0
    for k in dic1:
        Nk = len(dic1[k])
        s1 = set(dic1[k])
        for kk in dic2:
            s2 = set(dic2[kk])
            #olap = len(s1.intersection(s2))/float(Nk)
            olap = len(s1.intersection(s2))
            if olap > 0:
                df.loc[cnt] = ['a'+str(k),'b'+str(kk),olap]
                cnt = cnt+1
    return df

def dic2df(dic):
    rtn = {k:dic[k].copy() for k in dic}
    keys = [x for x in dic]
    lenmx = 0
    for k in keys:
        if len(dic[k])>lenmx:
            lenmx = len(dic[k])
            kmx = k
    for k in keys:
        if len(dic[k])<lenmx:
            for _ in range(lenmx-len(dic[k])):
                rtn[k].append('')
    return pd.DataFrame.from_dict(rtn)
    #return rtn

def dic_invert(d):
    inv = {}
    for k, v in d.items():
        if isinstance(v,list):
            for vv in v:
                keys = inv.setdefault(vv, [])
                keys.append(k)                
        else:
            keys = inv.setdefault(v, [])
            keys.append(k)
    for k in inv:
        if len(inv[k]) == 1:
            inv[k] = inv[k][0]
    return inv

def sprint(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents

def sprintdic(dic,chosen_country):
    global chosen_class
    chosen_class = None
    for label in dic:
        if chosen_country in dic[label]:
            chosen_class = label
            break

    rtn = ''
    if chosen_class == None:
        #print('Error: chosen_country not classified')
        rtn + sprint('Unclassified selection')
    elif chosen_class == -1:
        rtn = rtn + sprint('unclustered:')
    else:
        rtn = rtn + sprint('class '+str(chosen_class)+':')
    if chosen_class is not None:    
        countries = np.sort(np.array(dic[chosen_class]))
    else:
        print("Error sprintdic: no countries in class",chosen_class)
        return('')

    colwid = max([len(cc) for cc in countries[::2]]) + 5  # padding

    for i in range(0,len(countries),2):
        if i < len(countries)-1:
            rtn = rtn + sprint(countries[i].ljust(colwid)+countries[i+1])
            # rtn = rtn + sprint("".join(country.ljust(colwid) for country in [countries[i],countries[i+1]]))
        else:
            rtn = rtn + sprint(countries[i])
    return rtn

class Consensus:
    def __init__(self,
                 cldata,
                 cases = ['deaths', 'cases', 'cases_lin2020', 'cases_pwlfit', 'cases_nonlin', 'cases_nonlinr'],
                 ncomp = range(2,16),
                 minc = range(3,10),
                 min_samples = range(2,3), # 1 element [2] by default
                 satthresh = 0.7           # metaparam for swizzle, toward 1 => more unclustered
                 ):
        for cc in cases:
            if cc not in ['deaths', 'cases', 'cases_lin2020', 'cases_pwlfit', 'cases_nonlin', 'cases_nonlinr']:
                print('cases can only be one of:')
                print(['deaths', 'cases', 'cases_lin2020', 'cases_pwlfit', 'cases_nonlin', 'cases_nonlinr'])
        self.cases = cases
        self.ncomp = ncomp
        self.minc = minc
        self.min_samples = min_samples
        self.countries = list(cldata.clusdata_all[cases[0]].keys()) # save countries in first data set as list
        self.satthresh = satthresh
        self.clusdata = None
        self.swcountries=None
        self.cldata=cldata

    def scan(self,diag=False,progress=True,name=''):
        countries = self.countries
        lc = len(self.cases)
        maxvalid = [None]*lc
        maxvalidval= 0.0
        maxvalidsc = [None]*lc
        maxvalidscval= 0.0
        minscore1 = [None]*lc
        minscore1val = 999.
        minscore2 = [None]*lc
        minscore2val = 999.
        self.report = [' ']*4*lc
        self.reportdata = [None]*4*lc
        # runlen = len(self.cldata.clusdata_all[self.cases[0]])
        runlen = len(countries)
        self.probdata=np.zeros((4*lc,runlen),dtype=float)
        self.outlierdata=np.zeros((4*lc,runlen),dtype=float)
        self.clusdata = np.zeros((4*lc,runlen),dtype=np.int64)
        self.info =  pd.DataFrame(columns=['type','minc','mins','ncomp','clustered','unclustered','validity','validitysc','score1','score2'])
        infomax =  pd.DataFrame(columns=['type','minc','mins','ncomp','clustered','unclustered','validity','validitysc','score1','score2'])
        cnt=0

        for ic,case in tqdm(list(enumerate(self.cases)), desc=name+'loop over cases' ,disable= not progress): # loop with progress bar instead of just looping over enumerate(cases)
        # for ic,case in enumerate(self.cases):
            data = self.cldata.clusdata_all[case]
            #dat = np.array([data[cc] for cc in data]).astype(float)
            dat = np.array([data[cc] for cc in countries]).astype(float)
            for i in range(len(dat)):   # normalize data
                mx = max(dat[i])
                dat[i] = [dd/mx for dd in dat[i]]
            dat_disc = skfda.representation.grid.FDataGrid(dat,list(range(len(dat[0]))))

            if diag:
                print('--------------------------',case,'-------------------------------')
            maxvalidval= 0.0
            maxvalidscval= 0.0
            minscore1val = 999.
            minscore2val = 999.
            for ncomp in self.ncomp:  # code will only work if reference value 2 included in range
                fpca_disc = FPCA(n_components=ncomp)
                fpca_disc.fit(dat_disc)
                foo = fpca_disc.transform(dat_disc)
                for min_samples in self.min_samples:
                    for minc in self.minc:
                        clusterer = hdbscan.HDBSCAN(min_cluster_size=minc,min_samples=min_samples)
                        labels = clusterer.fit_predict(foo)
                        
                        # nclus = len(set([x for x in labels if x>-1]))
                        labelset = np.unique(labels)
                        nclus = len(np.unique(labels)) 
                        if -1 in labelset:
                            nclus = nclus-1
                        # nunclustered = sum([1 for x in labels if x==-1])
                        nunclustered = np.count_nonzero(labels == -1)
                        # nclustered = sum([1 for x in labels if x>-1])
                        nclustered = len(labels) - nunclustered

                        try:
                            validity = hdbscan.validity.validity_index(foo, labels)
                            validity = max(validity,0.001)
                            validitysc = rescale(validity,ncomp) 
                            score1 = 1.0/validitysc + float(nunclustered)/5. + np.abs(float(nclus)-4.)/2.
                            score2 = float(nunclustered)*(4.+np.abs(nclus-4.))/(validitysc*20.)
                            if validity > maxvalidval:
                                maxvalidval = validity
                                maxvalid[ic] = [(minc,min_samples,ncomp,nclus,nclustered,nunclustered,validity,validitysc,score1,score2)]
                                self.probdata[ic*4,:] = clusterer.probabilities_[:]
                                self.outlierdata[ic*4,:] = clusterer.outlier_scores_[:]
                                self.clusdata[ic*4,:] = labels[:]
                                self.report[ic*4] = 'max normal validity: %13s,%2d,%3d,%3d,%3d,%5.2f' % (case,minc,ncomp,nclus,nunclustered,validitysc)
                                self.reportdata[ic*4] = (case,minc,ncomp,nclus,nunclustered,validity,validitysc,score1,score2)
                            if validitysc > maxvalidscval:
                                maxvalidscval = validitysc
                                maxvalidsc[ic] = [(minc,min_samples,ncomp,nclus,nclustered,nunclustered,validity,validitysc,score1,score2)]
                                self.probdata[ic*4+1,:] = clusterer.probabilities_[:]
                                self.outlierdata[ic*4+1,:] = clusterer.outlier_scores_[:]
                                self.clusdata[ic*4+1,:] = labels[:]
                                self.report[ic*4+1] = 'max scaled validity: %13s,%2d,%3d,%3d,%3d,%5.2f' % (case,minc,ncomp,nclus,nunclustered,validitysc)
                                self.reportdata[ic*4+1] = (case,minc,ncomp,nclus,nunclustered,validity,validitysc,score1,score2)
                            if score1 <  minscore1val:
                                minscore1val = score1
                                minscore1[ic] = [(minc,min_samples,ncomp,nclus,nclustered,nunclustered,validity,validitysc,score1,score2)]   
                                self.probdata[ic*4+2,:] = clusterer.probabilities_[:]
                                self.outlierdata[ic*4+2,:] = clusterer.outlier_scores_[:]
                                self.clusdata[ic*4+2,:] = labels[:]
                                self.report[ic*4+2] = 'min combined score1: %13s,%2d,%3d,%3d,%3d,%5.2f' % (case,minc,ncomp,nclus,nunclustered,validitysc)
                                self.reportdata[ic*4+2] = (case,minc,ncomp,nclus,nunclustered,validity,validitysc,score1,score2)
                            if score2 <  minscore2val:
                                minscore2val = score2
                                minscore2[ic] = [(minc,min_samples,ncomp,nclus,nclustered,nunclustered,validity,validitysc,score1,score2)]
                                self.probdata[ic*4+3,:] = clusterer.probabilities_[:]
                                self.outlierdata[ic*4+3,:] = clusterer.outlier_scores_[:]
                                self.clusdata[ic*4+3,:] = labels[:]
                                self.report[ic*4+3] = 'min combined score2: %13s,%2d,%3d,%3d,%3d,%5.2f' % (case,minc,ncomp,nclus,nunclustered,validitysc)
                                self.reportdata[ic*4+3] = (case,minc,ncomp,nclus,nunclustered,validity,validitysc,score1,score2)

                            if diag:
                                print('hdbscan: ',minc,'minc:  ',min_samples,'mins:  ',ncomp ,'FPCAcomponents:  ',
                                      nclus,'clusters;  ',
                                      nclustered,'clustered;  ',
                                      nunclustered,'unclustered; ','validity =',np.round(validity,5),'validitysc =',np.round(validitysc,5),
                                      'score1:',np.round(score1,3),'score2:',np.round(score2,3))
                        except:
                            validity=None
                            if diag:
                                print('hdbscan: ',minc,'minc:  ',min_samples,'mins:  ',ncomp ,'FPCAcomponents:  ',
                                      nclus,'clusters;  ',
                                      nclustered,'clustered;  ',nunclustered,'unclustered; ','validity =',validity)
                        self.info.loc[cnt] = [case,minc,min_samples,ncomp,nclustered,nunclustered,validity,validitysc,score1,score2]
                        cnt = cnt+1

                    if diag:
                        print('--------------------------')
        self.probdata2 = np.where(self.probdata==0.,self.outlierdata,self.probdata)
        if diag:
            print('---------------------------------------------------------')
            print('minc,min_samples,ncomp,nclus,nclustered,nunclustered,validity,validitysc,score1,score2')
            print('maxvalid ',maxvalid[ic])
            print('maxvalidsc ',maxvalidsc[ic])
            print('minscore1',minscore1[ic])
            print('minscore2',minscore2[ic])
            #print('making clusters...')
            #self.make_clusters()
            #print('swizzling')
            #self.swizzle()
        
    def plot_outliers(self):
        Nvars = len(self.cases)*4
        max_cols = lc
        max_rows = Nvars // max_cols
        if Nvars % max_cols:
            max_rows = max_rows+1
        fig,axes = plt.subplots(lc,4,figsize=(4*lc,36))
        for n in range(Nvars):
            i = n % max_rows
            j = int (n/max_rows)
            ax = axes[j,i]
            ax.scatter(range(len(self.outlierdata[0])),self.probdata[n],color='blue',alpha=0.3,s=40)   # blue
            ax.scatter(range(len(self.outlierdata[0])),1-self.outlierdata[n],color='red',alpha=0.3,s=20)  # red
            ax.set_xlabel('country')
            ax.set_title(self.report[n])

    def  make_clusters(self,
                      refclustering='auto', # # fiducial column; change here.
                      diag=True):
        if len(self.clusdata) == 0:
            print('must run a scan to define and fill clusdata first: starting scan')
            self.scan()
        countries = self.countries
        if diag:
            print(len(countries),'countries')
        if refclustering == 'auto' or refclustering >= 4*len(self.cases):
            nrep = len(self.reportdata)
            scores = [rep[7] for rep in self.reportdata[3:nrep:4]] # optimal score 2 subset of data reports 
            refclustering = np.argmin(np.array(scores))*4+3        # ref clustering is clustering with minimal score 2
            if diag:
                print('reference clustering (numbered from 0) is',refclustering)
            self.refclustering = refclustering
        else:
            self.refclustering = refclustering

        clus_argsort = np.lexsort((countries,self.clusdata[self.refclustering]))  # lexicographical sort of countries by reference clustering and name
        scountries = [countries[clus_argsort[i]] for i in range(len(countries))]  #  sorted country names as above
        self.scountries = scountries
        self.sidx = clus_argsort
        # self.sinvidx = 


        self.probdata_s = self.probdata2.copy()                                   # sort the cluster ids and probs to match scountries
        self.clusdata_s = self.clusdata.copy()
        self.dics = []
        for i in range(len(self.probdata2)):
            tmp = {}
            for j in range(len(scountries)):
                self.probdata_s[i,j] = self.probdata2[i,clus_argsort[j]]
                self.clusdata_s[i,j] = self.clusdata[i,clus_argsort[j]]
                tmp[scountries[j]] = self.clusdata_s[i,j]
            self.dics.append(tmp.copy())
        self.clusdata_rs = self.clusdata_s[self.refclustering,:] 
        """
        This is the basic cluster comparison.  It suffers from the independent ordering of clusters, which makes the colourings different in each column. 
        * In general, given the different number of clusters this is a nontrivial problem in graph matching. We adopt a two phase approach in what follows: 
        * first choose a reference column (here column `refclustering=1` (defined in a cell above), not zero) with a good differentiated clustering.
        * relabel the clusters in each other column with the colours of the best matching cluster in the reference column (`coldata_adj`)
        * then relabel the colours again in case of split clusters, with the hybrid colour of the source cluster colour in reference column and the destination colour (`coldata_adj2`)

        `coldata`, `coldata_adj` and `coldata_adj2` are 3-d matrices: rows labeled by countries, columns labeled by report string (from max scoring), and 3 values for RGB in z-dim.
        """                
        rawdata = np.transpose(self.probdata_s)
        cindex = np.transpose(self.clusdata_s) 
        ncols = len(set(self.clusdata.flatten()))
        if ncols>16:
            print('Error: currently only 16 different colours allowed', ncols )
        colors = np.array([[1,1,1],[1,0,0],[0,1,0],[0,0,1],[1.,1.,0.],[1.,0.,1.],[0.,1.,1.],[0.5,1.,0.],
                           [0,1,0.5],[0.5,0,1],[0.5,1,0.5],[0.3,0.7,0.5],[0.5,0.7,0.3],[0.7,0.5,0.3],[0.1,0.7,0.7],[0.7,0.1,0.7]]) # black,red,green,blue,yellow,cyan,magenta,...
        # colors = np.concatenate((colors,colors))
        self.basecolors = colors
        cluscols = np.transpose(colors[cindex[:,:]+1],(2,0,1)) # transpose to allow elementwise multiplication with rawdata with separate r,g,b
        self.coldata = np.transpose((cluscols+3*cluscols*rawdata)/4.,(1,2,0))   # transpose back to have colours as elements of 2D array
        
        coldata_c = self.coldata.copy()
        coldata_t = np.transpose(coldata_c,(1,0,2))
        clusa = self.clusdata_s[self.refclustering]
        ca = coldata_t[self.refclustering]
        for i in range(0,len(self.clusdata_s)):
            if i != self.refclustering:
                clusb = self.clusdata_s[i]
                cb = coldata_t[i]
                # newcolors_b = clust(clusa,clusb,ca,cb,True,False)  # only do phase 1 of cluster recolouring
                newcolors_b = clust_lsa(clusa,clusb,ca,cb,base_colors=colors,relabel=True,merge=False)
                coldata_t[i,:] = newcolors_b[:]
        self.coldata_adj = np.transpose(coldata_t,(1,0,2))         # sorted by countries in reference clustering sorted order

        coldata_c2 = self.coldata.copy()
        coldata_t2 = np.transpose(coldata_c2,(1,0,2))
        clusa = self.clusdata_s[self.refclustering]
        ca = coldata_t2[self.refclustering]
        for i in range(0,len(self.clusdata_s)):
            if i != self.refclustering:
                clusb = self.clusdata_s[i]
                cb = coldata_t2[i]
                # newcolors_b = clust(clusa,clusb,ca,cb,True,True)   # do phases 1 and 2 or cluster recolouring
                newcolors_b = clust_lsa(clusa,clusb,ca,cb,base_colors=colors,relabel=True,merge=True)
                coldata_t2[i,:] = newcolors_b[:]
        self.coldata_adj2 = np.transpose(coldata_t2,(1,0,2))       # sorted by countries in reference clustering sorted order


    def plot_stage(self,stage=1):
        scountries = self.scountries
        if stage not in [1,2,3]:
            print('Currently there are only stages 1, 2, 3')
            return
        if stage==1:
            coldat = self.coldata
        elif stage==2:
            coldat = self.coldata_adj
        elif stage==3:
            coldat = self.coldata_adj2
    
        fig,ax = plt.subplots(1,1,figsize=(15,20))
        img = ax.imshow(coldat)
        ax.set_yticks(range(len(scountries)))
        ax.set_yticklabels(scountries)
        ax.set_xticks(range(len(self.clusdata_s)))
        plt.setp(ax.get_xticklabels(), rotation='vertical', family='monospace')
        ax.set_xticklabels(self.report,rotation='vertical')
        # fig.colorbar(img)
        #plt.show()
        return fig

    def plot_all_stages(self):
        # the three stages of cluster alignment
        scountries = self.scountries
        fig,axes = plt.subplots(1,3,figsize=(20,20))

        ax = axes[0]
        img = ax.imshow(self.coldata)
        ax.set_yticks(range(len(scountries)))
        ax.set_yticklabels(scountries)
        ax.set_xticks(range(len(self.clusdata_s)))
        plt.setp(ax.get_xticklabels(), rotation='vertical', family='monospace')
        ax.set_xticklabels(self.report,rotation='vertical')

        ax = axes[1]
        img = ax.imshow(self.coldata_adj)
        ax.set_yticks(range(len(scountries)))
        ax.set_yticklabels(scountries)
        ax.set_xticks(range(len(self.clusdata_s)))
        plt.setp(ax.get_xticklabels(), rotation='vertical', family='monospace')
        ax.set_xticklabels(self.report,rotation='vertical')

        ax = axes[2]
        img = ax.imshow(self.coldata_adj2)
        ax.set_yticks(range(len(scountries)))
        ax.set_yticklabels(scountries)
        ax.set_xticks(range(len(self.clusdata_s)))
        plt.setp(ax.get_xticklabels(), rotation='vertical', family='monospace')
        ax.set_xticklabels(self.report,rotation='vertical')

        fig.tight_layout(pad=2.0)
        #plt.show()
        return fig        

    def swizzle(self,cols=None,satthresh=None):
        if satthresh == None:
            satthresh = self.satthresh
        # scountries = self.scountries   # s stands for sorted (by ref clustering and lexicographically)
        if cols==None:
            self.cols=list(range(4*len(self.cases)))
        else:
            self.cols = cols
        dic,classes,idx,rgblist,hsvdic = swizzle3(self.scountries,self.coldata_adj2,self.cols,self.refclustering,self.basecolors,self.clusdata_rs,satthresh = satthresh)
        #print(cols.idx)
        self.classes = classes                                           # already swizzle ordered by swizzle3
        self.swdat = np.array([self.coldata_adj2[i] for i in idx])       # swdat is swizzle3 sorted coldata_adj2
        self.swcountries = [self.scountries[i] for i in idx]             # swcountries is swizzle3 sorted scountries
        self.swdic = dic
        self.swidx = idx
        self.rgbdic = {cc:mpcolors.hsv_to_rgb(hsvdic[cc][1:4]) for cc in hsvdic} 
        self.hsvdic = hsvdic
        self.rgblist = rgblist                                           # already swizzle ordered by swizzle3 
        

    def plot_swiz(self):
        data = self.swdat.copy()
        fig,ax = plt.subplots(1,1,figsize=(10,24))
        if self.cols is not None:
            todel = list(set(range(data.shape[1])) - set(self.cols))
            data1 = np.delete(data,todel,1)
        else:
            data1 = data
        img = ax.imshow(data1)
        ax.set_yticks(range(len(self.swcountries)))
        labels = [cc+' %d' % self.classes[i] for i,cc in enumerate(self.swcountries)]
        # ax.set_yticklabels(self.swcountries)
        ax.set_yticklabels(labels)
        if self.cols is None:
            rep = self.report
        else:
            rep = [self.report[i] for i in self.cols]
        ax.set_xticks(range(len(rep)))
        plt.setp(ax.get_xticklabels(), rotation='vertical', family='monospace')
        ax.set_xticklabels(rep,rotation='vertical')
        # plt.show()
        return fig

    def plot_quantile(self,dtype,cluster='all',chosen_country=None,title=True):
        classdic = self.swdic
        if cluster == 'all':
            classes = list(classdic.keys())
        elif isinstance(cluster,list):
            classes = cluster
        elif cluster == 'own':
            for label in classdic:
                if chosen_country in classdic[label]:
                    classes= [label]
                    break
        else:
            classes = [cluster]
        clusdata_all = self.cldata.clusdata_all

        fig, ax = plt.subplots(1,len(classes),figsize=(len(classes)*5,5),squeeze=False)
        cnt = 0
        # print('classdic',classdic)
        for label in classes:   
            nelts = len(classdic[label])
            if chosen_country and chosen_country in classdic[label]:
                maxval=np.amax(clusdata_all[dtype][chosen_country])
                datchosen = np.maximum(clusdata_all[dtype][chosen_country]/maxval,0.)
            #dats = [[max(z/max(clusdata_all[dtype][cc]),0.) for z in clusdata_all[dtype][cc]] for cc in classdic[label] ]
            dats=np.zeros((len(classdic[label]),len(clusdata_all[dtype]['Germany'])),dtype=float)
            for i,cc in enumerate(classdic[label]):
                maxval=np.amax(clusdata_all[dtype][cc])
                dats[i,:] = np.maximum(clusdata_all[dtype][cc]/maxval,0.)
            dats = np.transpose(np.array(dats))
            pdats = [pd.Series(dat) for dat in dats]
            qdats = [[pdat.quantile(q) for q in [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]] for pdat in pdats]
            data = np.transpose(np.array(qdats))
            #print(np.shape(dats))
            #print(np.shape(data))
            # data = qdats
            x = range(len(data[0]))
            clrs = ['#f0f0f0','#c0c0c0','#505050','#303030','#00a0a0','#c0c000','#303030','#505050','#c0c0c0','#f0f0f0'] # clrs[0] not used
            for i in range(1,len(data)):
                ax[0,cnt].fill_between(x,data[i-1],data[i],alpha=0.8,color=clrs[i-1]);
            if label != -1 and title:
                ax[0,cnt].set_title('Class '+('%d' % label)+(' with %d ' % nelts)+'elts')
            elif title:
                ax[0,cnt].set_title('Unclustered'+(' with %d ' % nelts)+'elts')
            # ax[cnt].set_xticklabels("")  # x labels for cases_nonlinr!
            if cnt>0:
                ax[0,cnt].set_yticklabels("")
            if chosen_country and chosen_country in classdic[label]:
                #print(len(x),len(datchosen))
                if 'deaths' in dtype:
                    ax[0,cnt].plot(x,datchosen,alpha=1.0,color='red');
                else:
                    ax[0,cnt].plot(x,datchosen,alpha=1.0,color='green');
            cnt = cnt+1
        if title:
            plt.suptitle(dtype);
        return
        
    def make_map(self):
        global geog,geog1,clusters,geo_json_data
        cldata = self.cldata
        def load_data(url, filename, file_type):
            r = requests.get(url)
            with open(filename, 'w') as f:
                f.write(r.content.decode("utf-8"))
            with open(filename, 'r') as f:
                return file_type(f)

        url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
        country_shapes = f'{url}/world-countries.json'
        # Loading a json data structure with geo data using json.load: geo_json_data
        geo_json_data = load_data(country_shapes,'json',json.load);
        fname = country_shapes
        geog = gpd.read_file(fname)
        geog1 = gpd.read_file(fname)

        # self.clusalign_hsv = swizzleHSV(self.scountries,self.coldata_adj2,self.cols,self.refclustering)
        df0list = [[term]+list(self.hsvdic[term]) for term in self.hsvdic]
        df0 = pd.DataFrame(df0list, columns = ['name','cluster','hue','sat','val'])

        dflist = [[term]+[list(self.hsvdic[term])[0]]+[list(self.hsvdic[term])[1:]] for term in self.hsvdic]
        df = pd.DataFrame(dflist, columns = ['name','cluster','hsv'])

        #df.replace('United States', 'United States of America', inplace=True)
        df.replace('United States of America', "United States", inplace = True)
        #df.replace('Tanzania', "United Republic of Tanzania", inplace = True)
        #df.replace('Democratic Republic of Congo', "Democratic Republic of the Congo", inplace = True)
        #df.replace('Congo', "Republic of the Congo", inplace = True)
        #df.replace('Lao', "Laos", inplace = True)
        #df.replace('Syrian Arab Republic', "Syria", inplace = True)
        #df.replace('Serbia', "Republic of Serbia", inplace = True)
        #df.replace('Czechia', "Czech Republic", inplace = True)
        #df.replace('UAE', "United Arab Emirates", inplace = True)
        df.replace('USA', "United States", inplace = True)

        
        geog.name.replace('United States of America', 'United States', inplace=True)
        geog.name.replace('USA', 'United States', inplace=True)
        geog.name.replace('United Republic of Tanzania', 'Tanzania',  inplace = True)
        geog.name.replace('Republic of Serbia', 'Serbia',  inplace = True)
        # geog.name.replace("Democratic Republic of the Congo", 'Democratic Republic of Congo', inplace = True)
        # geog.name.replace("Republic of the Congo", 'Congo',  inplace = True)
        # geog.name.replace('Laos', "Lao", inplace = True)
        # geog.name.replace("Syria", 'Syrian Arab Republic', inplace = True)   
        # geog.name.replace("Czech Republic",'Czechia', inplace = True)
        # geog.name.replace( "United Arab Emirates",'UAE', inplace = True)

        replace_dic = {'USA':'United States','United States of America':'United States','United Republic of Tanzania': 'Tanzania','Republic of Serbia': 'Serbia'}

        geogclus=geog.merge(df,how='left',on='name')

        # https://stackoverflow.com/questions/944700/how-can-i-check-for-nan-values
        # cluster and hsv data for chloro_data which could be used by chloropleth as x

        clusters =  dict(zip(geogclus['id'].tolist(), geogclus['cluster'].tolist()))
        clusters = {cc: -2 if math.isnan(clusters[cc]) else clusters[cc] for cc in clusters.keys()}
        clusterbn =  dict(zip(geogclus['name'].tolist(), geogclus['cluster'].tolist()))
        clusterbn = {cc: -2 if math.isnan(clusterbn[cc]) else clusterbn[cc] for cc in clusterbn.keys()}
        hsvbn =  dict(zip(geogclus['name'].tolist(), geogclus['hsv'].tolist()))
        hsvbn = {cc: [0.,0.,1.] if not isinstance(hsvbn[cc],list) else hsvbn[cc]  for cc in hsvbn.keys()}

        # now add cluster and hsv properties to geo_json_data to allow flexible use
        for feature in geo_json_data['features']:
            # print(feature)
            properties = feature['properties']
            name = properties['name']
            if name in replace_dic:
                properties['name'] = replace_dic[name]
                name = properties['name']
            if name in clusterbn:
                properties['cluster']= clusterbn[name]
            else:
                properties['cluster']= -2
            if name in hsvbn:    
                properties['hsv']= hsvbn[name]
            else:
                properties['hsv']= [0.,0.,1.]
            #print(name,properties['hsv'])

        def rgb_to_str(rgb):
            if len(rgb) == 3:
                return '#%02x%02x%02x' % (int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255))
            elif len(rgb) == 4:
                return '#%02x%02x%02x%02x' % (int(rgb[3]*255),int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255))
            else:
                return '#000000'

        def fillopacity(feature,colormap,x):
            global chosen_country,current_country,country_display,display_countries
            global chosen_class,current_class,class_display,chosen_swdic
            n = feature['properties']['name']
            if n in chosen_swdic[current_class]:
                o=1.0
            else:
                o=0.6
            return o


        def colorfill(feature,colormap,x):            
            c = feature['properties']['cluster']
            h = feature['properties']['hsv'][0]
            s = feature['properties']['hsv'][1]
            v = feature['properties']['hsv'][2]
            if c == -1:
                t = 0.2
            else:
                t = 0.
            rgb = list(mpcolors.hsv_to_rgb([h,s,v]))
            # rgb.append(t)
            return rgb_to_str(rgb)

        def colorborder(feature,colormap,x):
            c = feature['properties']['cluster']
            h = feature['properties']['hsv'][0]
            s = feature['properties']['hsv'][1]
            v = feature['properties']['hsv'][2]
            if c == -2:
                rgb = list(mpcolors.hsv_to_rgb([0.,0.,1.]))
            elif c == -1:
                rgb = list(mpcolors.hsv_to_rgb([0.,0.,0.]))
            else:
                rgb = list(mpcolors.hsv_to_rgb([h,1.,1.]))
            # rgb.append(t)
            return rgb_to_str(rgb)

        style_function = lambda feature,colormap,x: {"weight":0.5, 
                                    # 'color':'black',
                                    #'fillColor':colormap(x['properties']['hue']), 
                                    'border_opacity':0.5,
                                    'border_color':'gray',
                                    'color': 'gray',
                                    # 'border_color':colorborder(feature,colormap,x),
                                    # 'color': colorborder(feature,colormap,x),
                                    'fillColor':colorfill(feature,colormap,x), 
                                    'fillOpacity':fillopacity(feature,colormap,x)}
                                    #'fillOpacity':0.7}

        def update_html(feature,  **kwargs):
            # print('debug name hsv cluster',feature['properties']['name'],feature['properties']['cluster'],feature['properties']['hsv'])
            html.value = '''
                <h3 style="font-size:12px"><b>{}</b></h3>
                <h4 style="font-size:10px">Cluster: {:2d} </h4> 
                <h4 style="font-size:10px">Cluster colour mix: {}</h4>
                <h4 style="font-size:10px">Prob in a cluster: {}</h4>
                <h4 style="font-size:10px">Prob this cluster: {}</h4>
            '''.format(feature['properties']['name'],
                       feature['properties']['cluster'],
                       #"%.3f %.3f %.3f" % tuple(feature['properties']['hsv']))
                       "%.3f" % feature['properties']['hsv'][0],
                       "%.3f" % feature['properties']['hsv'][1],
                       "%.3f" % feature['properties']['hsv'][2])

        def myplot(cons, dataname='deaths',country='Australia'):
            if country in cons.countries:
                plt.plot(cons.cldata.clusdata_all[dataname][country])

        def on_click(feature,  **kwargs):
            global chosen_country,current_country,country_display,display_countries
            global chosen_class,current_class,class_display,chosen_swdic
            chosen_country = feature['properties']['name']
            current_country = feature['properties']['name']
            if chosen_country in display_countries:
                country_display.children[1].value=chosen_country
                chosen_class = None
                current_class = None
                for cl in chosen_swdic:
                    if chosen_country in chosen_swdic[cl]:
                        chosen_class = cl;
                        current_class= cl;
                        break
                class_display.value=sprintdic(chosen_swdic,chosen_country)

        def getvalue(change):
            # make the new value available
            #future.set_result(change.new)
            #widget.unobserve(getvalue, value)
            global chosen_country,country_display,display_countries
            global chosen_class,class_display,chosen_swdic
            chosen_country = change.new['properties']['name']
            if chosen_country in display_countries:
                country_display.children[1].value=chosen_country
                chosen_class = None
                for cl in chosen_swdic:
                    if chosen_country in chosen_swdic[cl]:
                        chosen_class = cl;
                        break
                class_display.value=sprintdic(chosen_swdic,chosen_country)
         
        #print('now forming chloropleth layer')   
        layer = ipyleaflet.Choropleth(
            geo_data=geo_json_data,
            choro_data=clusters,
            colormap=linear.YlOrRd_04,
            #border_color='black',
            #style={'fillOpacity': 0.9, 'dashArray': '5, 5'},
            style_callback = style_function,
            hover_style = {'fillOpacity': 0.9, 'dashArray': '5, 5'})
        #print('now setting up country control')
        html = HTML('''Selected Country''')
        html.layout.margin = '0px 10px 10px 10px'
        control = ipyleaflet.WidgetControl(widget=html, position='bottomleft')
        control.max_height= 140
        control.max_width= 220
        #print('now implementing map')
        m = ipyleaflet.Map(center = (20,10), zoom = 2)
        m.add_layer(layer)

        m.add_control(control)
        layer.on_click(on_click)
        layer.on_hover(update_html)
        layer.observe(getvalue,'change')
        m.add_control(ipyleaflet.FullScreenControl())

        self.map = m        
    
