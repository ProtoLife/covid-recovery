
import numpy as np

#######################################################################
# # ClusterFit class

class ClusterFit:
    """
    container class for fitting PCA, clustering
    """
    def __init__(self,
                 data,           # could be deaths/cases, raw/adjusted
                 Npca = 10,
                 fft = None,    # optionally True to do PCA on Fourier transformed data
                 outfile = ''):
        self.Npca = Npca
        self.data = data
        self.outfile = outfile
        self.dat = np.array([data[cc] for cc in data])

        self.pca = PCA(Npca)
        if fft == 'fft' or fft == 'powfft':
            self.fftdat = np.fft.rfft(self.dat) # last axis by default
            self.nfft = len(self.fftdat[0])
            if fft == 'powfft':
                self.fftpow = np.square(np.abs(self.fftdat))
                for i in range(len(self.fftpow)): # normalize data ignoring DC component
                    mx = max(self.fftpow[i])
                    self.fftpow[i] = [dd/mx for dd in self.fftpow[i]]
                self.lfftpow = np.log(self.fftpow)
                # self.pca.fit(self.fftpow)
                self.fitted = self.pca.fit_transform(self.lfftpow)
                self.smoothed = self.pca.inverse_transform(self.fitted)
                self.fft = 'powfft'
            else: # 'fft'
                # consider scaling data from all countries to same max freq amplitude per country of fft 
                self.rfft =  np.concatenate((np.real(self.fftdat),np.imag(self.fftdat)),axis = 1) # concatenate along 2nd axis
                # self.pca.fit(self.rfft)
                maxvals = np.zeros(len(self.dat))
                dcvals = np.zeros(len(self.dat))
                for i in range(len(self.rfft)): # normalize data ignoring DC component, scaling data from all countries to same max freq amplitude per country
                    dcvals[i] = self.rfft[i,0] # ignore DC component
                    self.rfft[i,0] = 0.
                    mx = maxvals[i] = max(self.rfft[i])
                    # mx = maxvals[i] = 1.0
                    self.rfft[i] = [dd/mx for dd in self.rfft[i]]
                self.fitted = self.pca.fit_transform(self.rfft)
                self.rsmoothed = self.pca.inverse_transform(self.fitted)
                self.fftsmoothed = np.transpose(np.array([self.rsmoothed[:,k] + self.rsmoothed[:,self.nfft+k]*1j for k in range(self.nfft)], dtype=np.cdouble))
                for i in range(len(data)):
                    self.fftsmoothed[i,:] =  self.fftsmoothed[i,:]*maxvals[i]
                self.fftsmoothed[:,0] = dcvals
                self.smoothed = np.fft.irfft(self.fftsmoothed,len(self.dat[0]))
                self.fft = 'fft'
        else:
            for i in range(len(self.dat)):   # normalize data
                mx = max(self.dat[i])
                self.dat[i] = [dd/mx for dd in self.dat[i]]
            # self.pca.fit(self.dat)
            self.fitted = self.pca.fit_transform(self.dat)
            self.smoothed = self.pca.inverse_transform(self.fitted)
            self.nfft = 0
            self.fft = None

        #print('explained_variance_ratio:')
        #print('explained_variance_ratio_' in dir(self.pca))
        #print([x for x in dir(self.pca) if '__' not in x])
        #print(self.pca.explained_variance_ratio_)
        #print('singular values:')
        #print(self.pca.singular_values_)

    def plot_2components(self):
        plt.scatter(self.fitted[:,0],fitted[:,1]);

    def cluster_plot_all(self):
        max_cols=6
        max_rows=int(len(self.dat)/max_cols) + 1
        fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20,3.5*max_rows))
        if self.fft == 'powfft' or self.fft == 'fft':
            axes2 = np.array([[ax.twinx() for ax in axesrow] for axesrow in axes])         
        countries = [cc for cc in self.data]

        if len(self.clus_labels) == len(countries): 
            print('sorting countries according to cluster labels') 
            self.clus_argsort = np.lexsort((countries,self.clus_labels))
            scountries = [countries[self.clus_argsort[i]] for i in range(len(countries))]
        else:
            scountries = countries

        for id, countrycode  in enumerate(countries):
            row = id // max_cols
            col = id % max_cols
            if len(self.clus_labels) == len(countries):
                idx = self.clus_argsort[id]
            else:
                idx = id
            axes[row, col].plot(self.dat[idx])
            if self.fft == 'powfft':
                axes2[row, col].plot(self.smoothed[idx],color='red')
                # axes2[row, col].set_yscale('log') # not required, data is already logarithmic
            elif self.fft == 'fit':
                axes2[row, col].plot(self.smoothed[idx],color='orange')
            else:
                axes[row, col].plot(self.smoothed[idx])
            axes[row, col].set_title(countries[idx])
        for idx in range(len(countries),max_rows*max_cols):
            row = idx // max_cols
            col = idx % max_cols
            axes[row, col].axis("off")
            if self.fft == 'powfft':
                axes2[row, col].axis("off")
        #plt.subplots_adjust(wspace=.05, hspace=.05)
        if self.outfile != '':
            plt.savefig(self.outfile)
        plt.show()

    def hdbscan(self,min_size=4):
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
        tdat = self.dat
        self.clus_labels = self.clusterer.fit_predict(tdat)
        validity = hdbscan.validity.validity_index(tdat, self.clus_labels)
        print('cluster validity index =',validity)
        print('cluster validity of each cluster:')
        validity = hdbscan.validity.validity_index(tdat, self.clus_labels,per_cluster_scores=True)
        for i,v in enumerate(validity):
            print('cluster',self.clus_labels[i],'validity =',validity[i])
            

    def plot_fpca(self):
        dat_disc = skfda.representation.grid.FDataGrid(dat,list(range(len(dat[0]))))
        fpca_disc = FPCA(n_components=10)
        fpca_disc.fit(dat_disc)
        fpca_disc.components_.plot()        

    def hdbscan_fpca(self,min_size=4,min_samples=3,n_components=5,diag=True):
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size,min_samples=min_samples)
        dat_disc = skfda.representation.grid.FDataGrid(dat,list(range(len(dat[0]))))
        fpca_disc = FPCA(n_components=n_components)
        fpca_disc.fit(dat_disc)
        self.fpca_transform = fpca_disc.transform(dat_disc)
        self.clus_labels = self.clusterer.fit_predict(self.fpca_transform)
        if diag:
            try:
                validity = hdbscan.validity.validity_index(self.fpca_transform, self.clus_labels)
                labels = self.clus_labels
                print('hdbscan_min_clus=',min_size,':  ',n_components ,'FPCAcomponents:  ',
                      len(set([x for x in labels if x>-1])),'clusters;  ',
                      sum([1 for x in labels if x>-1]),'clustered;  ',sum([1 for x in labels if x==-1]),'unclustered; ','validity =',np.round(validity,3))
            except:
                validity=None
                labels = self.clus_labels
                print('hdbscan_min_clus=',min_size,':  ',n_components ,'FPCAcomponents:  ',
                  len(set([x for x in labels if x>-1])),'clusters;  ',
                  sum([1 for x in labels if x>-1]),'clustered;  ',sum([1 for x in labels if x==-1]),'unclustered; ','validity =',validity)        

    def hdbscan_pca(self,min_size=4):
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
        tdat = self.fitted
        print('shape of cluster data = ',tdat.shape)
        self.clus_labels = self.clusterer.fit_predict(tdat)
        validity = hdbscan.validity.validity_index(tdat, self.clus_labels)
        print('cluster validity index =',validity)
        print('cluster validity of each cluster:')
        validity = hdbscan.validity.validity_index(tdat, self.clus_labels,per_cluster_scores=True)
        for i,v in enumerate(validity):
            print('cluster',self.clus_labels[i],'validity =',validity[i])

    def umap(self,random_state=0,n_neighbors=10):
        self.um_fit = umap.UMAP(random_state=random_state,n_neighbors=n_neighbors).fit(self.fitted)
        self.um_dat = [self.um_fit.embedding_[:,i] for i in range(2)]

    def umap_cluster(self,random_state=0,min_size=4,diag=True,n_neighbors=10):
        self.um_fit = umap.UMAP(random_state=random_state,n_neighbors=n_neighbors).fit(self.fitted)
        self.um_dat = [self.um_fit.embedding_[:,i] for i in range(2)]
        tdat = np.transpose(self.um_dat)

        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
        self.clus_labels = self.clusterer.fit_predict(tdat)
        self.clus_probs = self.clusterer.probabilities_
        if diag:
            print('hdbscan found',len(set(self.clus_labels)),'clusters.')
        
    def umap_best_cluster(self,Nclus=3,Ntries=50,minsize=4,ranstate=0,n_neighbors=10):
        clusall = []
        clus = {}
        clus['probs'] = []
        clus['idx'] = []
        for i in range(ranstate,ranstate+Ntries):
            self.umap_cluster(random_state=i,min_size=minsize,diag=False,n_neighbors=n_neighbors)
            if len(set(self.clus_labels)) == Nclus:
                clus['probs'].append(np.mean(self.clus_probs))
                clus['idx'].append(i)
        print('found',len(clus['probs']),'clusterings with size',Nclus,'clusters')
        if len(clus['probs'])>1:
            idx = np.argsort(clus['probs'])[-1:][0]
        elif len(clus['probs']) == 1:
            idx = 0
        else:
            print("Failed to find a cluster with",Nclus,"components")
            return
        self.umap_cluster(random_state=clus['idx'][idx],min_size=minsize,diag=False,n_neighbors=n_neighbors)

    
    def plot_umap(self):
        labs = [x for x in self.clus_labels]
        for i in range(len(labs)):
            if labs[i]<0:
                labs[i] = None
        plt.scatter(self.um_dat[0],self.um_dat[1],c=labs)
        xx = [self.um_dat[0][i] for i in range(len(labs)) if labs[i]==None]
        yy = [self.um_dat[0][i] for i in range(len(labs)) if labs[i]==None]
        #print(xx)
        #print(yy)
        plt.scatter(xx,yy,color='red')   
        
    def plot_pcas(self):
        max_cols = 5
        max_rows = self.Npca // max_cols
        if self.Npca%max_cols>0:
            max_rows = max_rows+1
        fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20,max_rows*3.5))
        for i in range(10):
            foo = np.zeros(10)
            foo[i] = 1
            mypca = self.pca.inverse_transform(foo)
            if self.fft == 'fft':
                fftmypca = np.array([mypca[k] + mypca[self.nfft+k]*1j for k in range(self.nfft)], dtype=np.cdouble) 
                mypca = np.fft.irfft(fftmypca)
            row = i // max_cols
            col = i % max_cols
            #axes[row, col].axis("off")
            axes[row, col].plot(mypca)
           
