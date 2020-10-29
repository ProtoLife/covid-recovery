#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib import colors as mpcolors

import numpy as np
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


def clust(clustering_a,clustering_b,colors_a,colors_b): 
    """ relables clustering b to match clustering a
        if more than one cluster in a optimally matches a particular cluster in b, then color of b is merger of colors in a
        if more than one cluster in b optimally matches a particular cluster in a, then colors in a merged and split for b
    """
    labels_a = set(clustering_a)
    labels_b = set(clustering_b)
    
    if len(labels_a) != len(colors_a): print('error wrong color list length for a')
    if len(labels_b) != len(colors_b): print('error wrong color list length for b')
            
    a_to_b = {}
    b_to_a = {}
    a_cols = {a : colors_a[i] for i,a in enumerate(labels_a)}
    b_cols = {b : colors_b[i] for i,b in enumerate(labels_b)}
    
    for a in labels_a:
        maxscore = 0
        maxlab = -2
        for b in labels_b:
            score = score_int(matchset(clustering_a,a),matchset(clustering_b,b))
            if score > maxscore:
                maxscore = score
                maxlab = b
        a_to_b.update({a:maxlab})

    for b in labels_b:
        maxscore = 0
        maxlab = -2
        for a in labels_a:
            score = score_int(matchset(clustering_a,a),matchset(clustering_b,b))
            if score > maxscore:
                maxscore = score
                maxlab = a
        b_to_a.update({b:maxlab})
    
    for b in labels_b:   # first adjust colors in b to match mapped clusters from a (transfer and merge)
        amap = [a for a in labels_a if a_to_b[a] == b]
        if len(amap) > 0:
            h = sum([mpcolors.rgb_to_hsv(a_cols[a])[0] for a in amap])/len(amap) # average hue from amap
            s = mpcolors.rgb_to_hsv(b_cols[b])[1] # take s saturation from b
            v = mpcolors.rgb_to_hsv(b_cols[b])[2] # take v from b
            b_cols[b] = mpcolors.hsv_to_rgb([h,s,v]) # back to rgb

    for a in labels_a:   # now readjust colors in b that both map to same a (split)
        bmap = [b for b in labels_b if b_to_a[b] == a]
        if len(bmap)>1:
            h = sum([mpcolors.rgb_to_hsv(b_cols[b])[0] for b in bmap])/len(bmap) # average hue from bmap  
            ha = mpcolors.rgb_to_hsv(a_cols[a])[0]
            hb = np.linspace(abs(h-ha/4.),abs(h+ha/4.),len(bmap))
            #print('hb[',hb[0],hb[1],']',h,ha,ha/4.,abs(h-ha/4.),abs(h+ha/4.))
            for i,b in enumerate(bmap):
                s = mpcolors.rgb_to_hsv(b_cols[b])[1] # take s saturation from b
                #print('s',s)
                v = mpcolors.rgb_to_hsv(b_cols[b])[2] # take v from b
                #print('v',v)
                b_cols[b]= mpcolors.hsv_to_rgb([hb[i],s,v])
                #print('hb[i],b_cols[b]',hb[i],b_cols[b])
    return b_cols,a_to_b,b_to_a

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
        hmaxcnt = np.argmax(np.array(hcnts)) # problem if hcnts is empty sequence
    if modal and hcnts[hmaxcnt] >= len(rgb_colours)/4:
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
                      
def clust(clustering_a,clustering_b,colors_a,colors_b,relabel=True,merge=True): 
    """ relables clustering b to match clustering a
        if more than one cluster in a optimally matches a particular cluster in b, then color of b is merger of colors in a
        if more than one cluster in b optimally matches a particular cluster in a, then colors in a merged and split for b
    """
    labels_a = list(set(clustering_a))
    labels_b = list(set(clustering_b))
    newcolors_b = np.zeros((len(colors_b),3),dtype=float)
    newcolors_b[:,:] = colors_b[:,:]
            
    a_to_b = {}
    b_to_a = {}
    a_cols = {}
    b_cols = {}
    
    for a in labels_a:
        maxscore = 0
        maxlab = -2
        for b in labels_b:
            score = score_int_union(matchset(clustering_a,a),matchset(clustering_b,b))
            if score > maxscore:
                maxscore = score
                maxlab = b
        a_to_b.update({a:(maxlab,maxscore)})
    maxvals_a_to_b = [a_to_b[a][1] for a in labels_a]
    reorder_a = np.flip(np.argsort(maxvals_a_to_b))
    labels_a_sort = [labels_a[r] for r in list(reorder_a)]

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

    if relabel:    
        for b in labels_b_sort:   # first adjust colors_b to match mapped clusters from a (transfer and merge)
            amap = [a for a in labels_a_sort if a_to_b[a][0] == b]
            for a in amap:
                alist = matchset(clustering_a,a)
                a_cols[a] = colors_a[alist[0]]
            blist = matchset(clustering_b,b)
            amap_t = list(set(amap)-set([-1]))
            if len(amap_t) > 0: # some non-unclustered (ie not -1) clusters in a map to b
                # h = sum([mpcolors.rgb_to_hsv(a_cols[a])[0] for a in amap])/len(amap) # average hue from amap
                h = color_mean_rgb_to_hsv([a_cols[a] for a in amap_t],[a_to_b[a][1] for a in amap_t])[0]
                for j in blist:
                    s = mpcolors.rgb_to_hsv(colors_b[j])[1] # take s saturation from b
                    v = mpcolors.rgb_to_hsv(colors_b[j])[2] # take v from b
                    newcolors_b[j,:] = mpcolors.hsv_to_rgb([h,s,v]) # back to rgb  
            b_cols[b] = newcolors_b[blist[0]] # first matching elt colour (to extract hue)
            
    if merge:
        for a in labels_a_sort:   # now readjust colors in b that both map to same a (split)
            bmap = [b for b in labels_b_sort if b_to_a[b][0] == a]
            if len(bmap)>1:                          
                for i,b in enumerate(bmap):
                    blist = matchset(clustering_b,b)
                    # h = (mpcolors.rgb_to_hsv(b_cols[b])[0] + mpcolors.rgb_to_hsv(a_cols[a])[0])/2
                    h = color_mean_rgb_to_hsv([b_cols[b],a_cols[a]])[0]
                    for j in blist:                     
                        s = mpcolors.rgb_to_hsv(b_cols[b])[1] # take s saturation from b
                        v = mpcolors.rgb_to_hsv(b_cols[b])[2] # take v from b
                        newcolors_b[j,:]= mpcolors.hsv_to_rgb([h,s,v])

    return newcolors_b

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
    plt.show()


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


def swizzle(countries,data,cols):
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


def swizzle3(countries,data,cols,refcol,satthresh = 0.7):
    eps = 0.0001
    clus = [None]*len(countries)
    rgblist = [None]*len(countries)
    hsvdic = {}
    hsvrefs = [mpcolors.rgb_to_hsv(c) for c in data[:,refcol]]
    huesref  = np.sort(list(set([hsv[0] for hsv in hsvrefs if hsv[1] > eps])))
    # print('huesref',huesref)
    for i in range(len(countries)):
        # hsvsc = hscore_org(data[i,:,:],cols)
        # print(countries[i])
        # hsvsc = hscore_mode_org(data[i,:,:],cols) # using modal hue
        hsvsc = hscore_org(data[i,:,:],cols)      # using color circle mean hue
        hue = hsvsc[0]
        sat = hsvsc[1]
        if sat <= satthresh:  # mean is classed as unclustered
            clus[i] = -1
        else:
            clus[i] = closest_hue(hue,huesref)
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


class Consensus:
    def __init__(self,
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
        self.countries = list(clusdata_all[cases[0]].keys()) # save countries in first data set as list
        self.satthresh = satthresh
        self.clusdata = None
        self.swcountries=None

    def scan(self,diag=False):
        countries = self.countries
        maxvalid = [None,None,None,None,None,None]
        maxvalidval= 0.0
        maxvalidsc = [None,None,None,None,None,None]
        maxvalidscval= 0.0
        minscore1 = [None,None,None,None,None,None]
        minscore1val = 999.
        minscore2 = [None,None,None,None,None,None]
        minscore2val = 999.
        self.report = [' ']*4*len(self.cases)
        self.reportdata = [None]*4*len(self.cases)
        runlen = len(clusdata_all[cases[0]])
        self.probdata=np.zeros((4*6,runlen),dtype=float)
        self.outlierdata=np.zeros((4*6,runlen),dtype=float)
        self.clusdata = np.zeros((4*6,len(countries)),dtype=np.int64)
        self.info =  pd.DataFrame(columns=['type','minc','mins','ncomp','clustered','unclustered','validity','validitysc','score1','score2'])
        infomax =  pd.DataFrame(columns=['type','minc','mins','ncomp','clustered','unclustered','validity','validitysc','score1','score2'])
        cnt=0

        for ic,case in tqdm(list(enumerate(self.cases)), desc='loop over cases' ,disable= not diag): # loop with progress bar instead of just looping over enumerate(cases)
        # for ic,case in enumerate(cases):
            data = clusdata_all[case]
            dat = np.array([data[cc] for cc in data]).astype(float)
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
            print('making clusters...')
            self.make_clusters()
            print('swizzling')
            self.swizzle()
        
    def plot_outliers(self):
        Nvars = len(self.cases)*4
        max_cols = 6
        max_rows = Nvars // max_cols
        if Nvars % max_cols:
            max_rows = max_rows+1
        fig,axes = plt.subplots(6,4,figsize=(24,36))
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
            print('currently only 16 different colours allowed', ncols )
        colors = np.array([[1,1,1],[1,0,0],[0,1,0],[0,0,1],[1.,1.,0.],[1.,0.,1.],[0.,1.,1.],[0.5,1.,0.],
                           [0,1,0.5],[0.5,0,1],[0.5,1,0.5],[0.3,0.7,0.5],[0.5,0.7,0.3],[0.7,0.5,0.3],[0.1,0.7,0.7],[0.7,0.1,0.7]]) # black,red,green,blue,yellow,cyan,magenta,...
        colors = np.concatenate((colors,colors))
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
                newcolors_b = clust(clusa,clusb,ca,cb,True,False)  # only do phase 1 of cluster recolouring
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
                newcolors_b = clust(clusa,clusb,ca,cb,True,True)   # do phases 1 and 2 or cluster recolouring
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
        plt.show()
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
        plt.show()
        return fig        

    def swizzle(self,cols=None,satthresh=None):
        if satthresh == None:
            satthresh = self.satthresh
        # scountries = self.scountries   # s stands for sorted (by ref clustering and lexicographically)
        if cols==None:
            self.cols=list(range(4*len(self.cases)))
        else:
            self.cols = cols
        dic,classes,idx,rgblist,hsvdic = swizzle3(self.scountries,self.coldata_adj2,self.cols,self.refclustering,satthresh = satthresh)
        #print(cols.idx)
        self.classes = classes                                           # already swizzle ordered by swizzle3
        self.swdat = np.array([self.coldata_adj2[i] for i in idx])       # swdat is swizzle3 sorted coldata_adj2
        self.swcountries = [self.scountries[i] for i in idx]             # swcountries is swizzle3 sorted scountries
        self.swdic = dic
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
        plt.show()
        return fig
        
    def make_map(self):
        global geog,geog1,clusters,geo_json_data
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
                                    'border_color':colorborder(feature,colormap,x),
                                    'color': colorborder(feature,colormap,x),
                                    'fillColor':colorfill(feature,colormap,x), 
                                    'fillOpacity':0.8}

        def update_html(feature,  **kwargs):
            global chosen_country
            global country_display
            chosen_country = feature['properties']['name']
            if country_display:
                country_display.children[1].value=chosen_country


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

        def myplot(dataname='deaths',country='Australia'):
            plt.plot(clusdata_all[dataname][country])

        def on_clicked(feature,  **kwargs):
            global chosen_country
            global country_display
            chosen_country = feature['properties']['name']
            country_display.children[1].value=chosen_country
            print("Country selected",chosen_country)

        def getvalue(change):
            # make the new value available
            #future.set_result(change.new)
            #widget.unobserve(getvalue, value)
            global chosen_country,chosen_country_widget
            chosen_country = change.new['properties']['name']
            print("Country selected is",chosen_country)
         
        #print('now forming chloropleth layer')   
        layer = ipyleaflet.Choropleth(
            geo_data=geo_json_data,
            choro_data=clusters,
            colormap=linear.YlOrRd_04,
            #border_color='black',
            #style={'fillOpacity': 0.9, 'dashArray': '5, 5'},
            style_callback = style_function)
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
        layer.on_hover(update_html)
        layer.on_click(on_clicked)
        layer.observe(getvalue,'change')
        m.add_control(ipyleaflet.FullScreenControl())

        self.map = m        
    
