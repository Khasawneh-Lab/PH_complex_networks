"""
PHN.py 
====================================
This Persistent Homology of Networks PHN script is used to form a complex network from either ordinal partitions (permutations) or k nearest neighbors.
"""


def FNN_dim(tau,ts,Rtol=None,Atol=None):
    """This function returns a suitable embedding dimension, given a time series and embedding delay, based on the false nearest neighbors algorithm.

    Args:
       tau (int):  The time delay for time series reconstrctuion (From mutual information function).
       ts (array):  Time series array (1d).

    Kwargs:
       Rtol (float): Tolerance for FNN function.
       Atol (float): Tolerance for FNN function.

    Returns:
       (int): n, The embedding dimension for delay embedding reconstruction.

    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from scipy import sparse
    from ripser import ripser
    
    minDim,maxDim = 3, 8
    #if no values entered for Rtol and Atol, use default ones
    if Rtol==None:
        Rtol=5
    if Atol==None:
        Atol=2
    if len(ts)-(maxDim-1)*tau < 20:
        maxDim=len(ts)-(maxDim-1)*tau-1
    ts = np.reshape(ts, (len(ts),1)) #ts is a column vector
    st_dev=np.std(ts) #standart deviation of the time series
    
    ndim=max(1,maxDim-minDim+1) #of dimensions to test
    Xfnn=np.zeros((1,ndim))
    for i in range(1,ndim+2):
        dim=i
        #delay reconstruction
        xlen = len(ts)-(dim-1)*tau
        a= np.linspace(0,xlen-1,xlen)
        a= np.reshape(a,(xlen,1))
        delayVec=np.linspace(0,(dim-1),dim)*tau
        delayVec= np.reshape(delayVec,(1,dim))
        delayMat=np.tile(delayVec,(xlen,1))
        vec=np.tile(a,(1,dim))
        indRecon = np.reshape(vec,(xlen,dim)) + delayMat
        indRecon = indRecon.astype(np.int64)
        tsrecon = ts[indRecon]
        tsrecon = tsrecon[:,:,0]
        
        from scipy.spatial import KDTree
        tree=KDTree(tsrecon)
        D,IDX=tree.query(tsrecon,k=2)
        
        #Calculate the false nearest neighbor ratio for each dimension
        if i>1:
            D_mp1=np.sqrt(np.sum((np.square(tsrecon[ind_m,:]-tsrecon[ind,:])),axis=1))
            #Criteria 1 : increase in distance between neighbors is large
            num1 = np.heaviside(np.divide(abs(tsrecon[ind_m,-1]-tsrecon[ind,-1]),Dm)-Rtol,0.5)
            #Criteria 2 : nearest neighbor not necessarily close to y(n)
            num2= np.heaviside(Atol-D_mp1/st_dev,0.5)
            num=sum(np.multiply(num1,num2))
            den=sum(num2)
            Xfnn[0,i-2]=num/den*100
        # Save the index to D and k(n) in dimension m for comparison with the
        # same distance in m+1 dimension   
        xlen2=len(ts)-dim*tau
        Dm=D[0:xlen2,-1]
        ind_m=IDX[0:xlen2,-1]
        ind=ind_m<=xlen2-1
        ind_m=ind_m[ind]
        Dm=Dm[ind]
        
    Number_FNN = Xfnn[0]
    for i in range(0, len(Number_FNN)):
        if Number_FNN[i]<Rtol:
            dim=i+1
            break
    return dim  





class Partitions:
    def __init__(self, data = None,
                 meshingScheme = None,
                 numParts=3,
                 alpha=0.05):
        import scipy

        if data is not None:
            # check that the data is in ordinal coordinates
            if not self.isOrdinal(data):
                print("Converting the data to ordinal...")
                # perform ordinal sampling (ranking) transformation
                xRanked = scipy.stats.rankdata(data[:,0], method='ordinal')
                yRanked = scipy.stats.rankdata(data[:,1], method='ordinal')


                xFloats = np.copy(data[:,0])
                xFloats.sort()
                yFloats = np.copy(data[:,1])
                yFloats.sort()

                self.xFloats = xFloats
                self.yFloats = yFloats


                data = np.column_stack((xRanked,yRanked))

                # and return an empty partition bucket

            # If there is data, set the bounding box to be the max and min in the data


            xmin = data[:,0].min()
            xmax = data[:,0].max()
            ymin = data[:,1].min()
            ymax = data[:,1].max()

            self.borders = {}
            self.borders['nodes'] = np.array([xmin, xmax, ymin, ymax])
            self.borders['npts'] = data.shape[0]
            self.numParts = numParts
            self.alpha = alpha



            # If there is data, use the chosen meshing scheme to build the partitions.
            if meshingScheme == 'DV' and self.isOrdinal(data):
                # Figure out
                self.partitionBucket = self.return_partition_DV(data = data,
                                        borders = self.borders,
                                        r = self.numParts,
                                        alpha = self.alpha)
            else: # meshingScheme == None
            # Note that right now, this will just do the dumb thing for every other input
                self.partitionBucket = [self.borders]
                #  set the partitions to just be the bounding box


        else:
            self.partitionBucket = []

    def convertOrdToFloat(self,partitionEntry):
        bdyList = partitionEntry['nodes'].copy()
        # Need to subtract one to deal with counting from
        # 0 vs counting from 1 problems
        xLow = int(bdyList[0])-1
        xHigh = int(bdyList[1])-1
        yLow = int(bdyList[2])-1
        yHigh = int(bdyList[3])-1


        if hasattr(self, 'xFloats'):
            xLowFloat = self.xFloats[xLow]
            xHighFloat= self.xFloats[xHigh]
            yLowFloat = self.yFloats[yLow]
            yHighFloat = self.yFloats[yHigh]
            convertedBdyList = [xLowFloat, xHighFloat, yLowFloat,yHighFloat]
            partitionEntry['nodes'] = convertedBdyList
            return partitionEntry
        else:
            print("You're trying to convert your ordinal data")
            print("back to floats, but you must have had ordinal")
            print("to begin with so I can't.  Exiting...")

    def __len__(self):
        return len(self.partitionBucket)

    def __getitem__(self,key):
        if hasattr(self,'xFloats'): #if the data wasn't ordinal
            entry = self.partitionBucket[key].copy()
            entry = self.convertOrdToFloat(entry)
            return entry
        else:
            return self.partitionBucket[key]

    def getOrdinal(self,key):
        # overrides the builtin magic method in the case where
        # you had non-ordinal data but still want the ordinal
        # stuff back.
        # If the data wasn't ordinal, this has the exact same
        # effect as self[key]
        return self.partitionBucket[key]

    def __iter__(self):
        # iterates over the converted entries in the
        # parameter bucket
        if hasattr(self,'xFloats'):
            return map( self.convertOrdToFloat, deepcopy(self.partitionBucket)  )
        else:
            return iter(self.partitionBucket)

    def iterOrdinal(self):
        # functions just like iter magic method without
        # converting each entry back to its float
        return iter(self.partitionBucket)

    def __str__(self):
        attrs = vars(self)
        output = ''
        output += 'Variables in partition bucket\n'
        output += '---\n'
        for key in attrs.keys():
            output += str(key) + ' : '
            output += str(attrs[key])+ '\n'
            output += '---\n'
        return output

    def plot(self):
        import matplotlib.pyplot as plt
        import matplotlib
        # plot the partitions
        fig1, ax1 = plt.subplots()
        for binNode in self:
            # print(binNode)
            # get the bottom left corner
            corner = (binNode['nodes'][0], binNode['nodes'][2])

            # get the width and height
            width = binNode['nodes'][1] - binNode['nodes'][0]
            height = binNode['nodes'][3] - binNode['nodes'][2]

            # add the corresponding rectangle
            ax1.add_patch(matplotlib.patches.Rectangle(corner, width, height, fill=False))

        # Doesn't show unless we do this
        plt.axis('tight')

    # helper function for error checking. Used to make sure input is in
    # ordinarl coordinates. It checks that when the two data columns are sorted
    # they are each equal to an ordered vector with the same number of rows.
    def isOrdinal(self, dd):
        return np.all(np.equal(np.sort(dd, axis=0),
                        np.reshape(np.repeat(np.arange(start=1,stop=dd.shape[0]+1),
                                             2), (dd.shape[0], 2))))




    # data: is a manyx2 numpy array that contains all the original data
    # borders: a dictionary that contains 'nodes' with a numpyu array of Xmin, Xmax, Ymin, Ymax,
    # and 'npts' which contains the number of points in the bin
    # r: is the number of partitions
    # alpha: the significance level to test for independence
    def return_partition_DV(self, data, borders, r=2, alpha=0.05):
        import scipy
        # extract the bin boundaries
        Xmin = borders['nodes'][0]
        Xmax = borders['nodes'][1]
        Ymin = borders['nodes'][2]
        Ymax = borders['nodes'][3]

        # find the number of bins
    #    numBins = r ** 2
        idx = np.where((data[:, 0] >= Xmin)
                       & (data[:, 0] <= Xmax )
                       & (data[:, 1] >= Ymin)
                       & (data[:, 1] <= Ymax))

        # extract the points in the bin
        Xsub = data[idx, 0]
        Ysub = data[idx, 1]

    #    print(Xsub.shape, '\t', Ysub.shape)

        # find the indices of the points in the x- and y-patches
        idx_x = np.where((data[:, 0] >= Xmin) & (data[:, 0] <= Xmax))
        idx_y = np.where((data[:, 1] >= Ymin) & (data[:, 1] <= Ymax))

        # get the subpartitions
        ai = np.floor(np.percentile(data[idx_x, 0], 1/r * np.arange(1, r) * 100))
        bj = np.floor(np.percentile(data[idx_y, 1], 1/r * np.arange(1, r) * 100))

        # get the bin edges
        edges1 = np.concatenate(([Xmin], ai, [Xmax]))
        edges2 = np.concatenate(([Ymin], bj, [Ymax]))

        # first exit criteria: we cannot split inot unique boundaries any more
        # preallocate the partition list
        partitions = []
        if (len(np.unique(edges1, return_counts=True)[1]) < r + 1 or
             len(np.unique(edges2, return_counts=True)[1])< r + 1):

            # reject futher partitions, and return original bin
            partitions.insert(0, {'nodes': np.array([Xmin, Xmax, Ymin, Ymax]),
                      'npts': len(idx[0])})
            return partitions

        # figure out the shift in the edges so that boundaries do not overlap
        xShift = np.zeros( (2 * r, 2 * r))
        yShift = xShift
        xShift[:, 1:-1] = np.tile(np.array([[-1, 0]]), (2 * r, r - 1))
        yShift = xShift.T

        # find the boundaries for each bin
        # duplicate inner nodes for x mesh
        dupMidNodesX = np.append(np.insert(np.repeat((edges1[1:-1]), 2, axis=0),
                                          0, edges1[0]), edges1[-1])

        # duplicate inner nodes for y mesh
        dupMidNodesY = np.append(np.insert(np.repeat((edges2[1:-1]), 2, axis=0),
                                          0, edges2[0]), edges2[-1])
        # reshape
        dupMidNodesY = np.reshape(dupMidNodesY, (-1, 1))

        # now find the nodes for each bin
        xBinBound = dupMidNodesX + xShift
        yBinBound = dupMidNodesY + yShift

        # find the number of points in each bin, and put this info into array
        binned_data = scipy.stats.binned_statistic_2d(Xsub.flatten(), Ysub.flatten(), None, 'count',
                                          bins=[edges1, edges2])
        # get the counts. Flatten columnwise to match the bin definition in the
        # loop that creates the dictionaries below
        binCounts = binned_data.statistic.flatten('F')

        # define an empty list to hold the dictionaries of the fresh partitions
        bins = []
        # create dictionaries for each bin
        # start with the loop over y
        # note how the loop counts were obtained above to match the convention
        # here
        for yInd in np.arange(r):
            # this is the loop over x
            for xInd in np.arange(r):
                # get the bin number
                binNo = yInd * r  + xInd
                xLow, xHigh = xBinBound[yInd, 2*xInd + np.arange(2)]
                yLow, yHigh = yBinBound[2*yInd + np.arange(2), xInd]
                bins.append({'nodes': np.array([xLow, xHigh, yLow, yHigh]),
                    'npts': binCounts[binNo] })

        # calculate the chi square statistic
        chi2 = scipy.stats.chisquare(binCounts)

        # check for independence and start recursion
        # if the chi2 test fails, do further partitioning:
        if (chi2.pvalue < alpha and Xmax!=Xmin and Ymax!=Ymin).all():
            for binInfo in bins:
                if binInfo['npts'] !=0:  # if the bin is not empty:
                    # append entries to the tuple
                    partitions.extend(self.return_partition_DV(data=data,
                                                            borders=binInfo,
                                                            r=r, alpha=alpha))

        # Second exit criteria:
        # if the partitions are independent, reject further partitioning and
        # save the orignal, unpartitioned bin
        elif len(idx[0]) !=0:
            partitions.insert(0, {'nodes': np.array([Xmin, Xmax, Ymin, Ymax]),
                      'npts': len(idx[0])})

        return partitions


# This function computes the mutual information function based on the
# algorithm described in:
# "Estimation of information by an adaptive partitioning of the observation
# space," Georges Darbellay and Igor Vajda, 1999.

def mutualInfo(x, y): 
   
    #input: x: x cooridnate array (could be time array), y: y coorindate array (could be time series)
    import numpy as np
    from scipy.stats import rankdata
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from scipy import sparse
    from ripser import ripser

    # perform ordinal sampling (ranking) transformation
    xRanked = rankdata(x, method='ordinal')
    yRanked = rankdata(y, method='ordinal')

    # obtain the adaptive mesh
    numParts = 4

    # get the adaptive partition of the data
    partitionList = Partitions(np.column_stack((xRanked, yRanked)), meshingScheme = "DV", numParts=numParts)
    #partitions are sorted in class as:
    #1. overall borders with number of points total: xmin, xmax, ymin, ymax
    #2. numParts = max number of parts in a single bin/partition
    #3. alpha
    #4. partition bucket: this has all the partition's borders and number of points in each.


    # extract the bin counts
    binCounts = np.array([partitionList.partitionBucket[i].get("npts") for i in range(len(partitionList))])

    # get the total number of points
    N = partitionList.borders.get("npts")
    
    # grab the probability information from the partition
    Pn_AB = binCounts / N

    # grab the probbility of the horizontal strips for each bin
    PC = partitionList #partition cells
    
    Pn_AR = np.zeros(len(partitionList))
    Pn_RB = np.zeros(len(partitionList))
    
    for Bin in range(len(partitionList)): #go through each bin
        Pn_AR[Bin] = len([xRanked for i in range(N) #count number of point between x bounds of bin
            if xRanked[i] >= PC[Bin].get('nodes')[0] and xRanked[i] <= PC[Bin].get('nodes')[1]])
        Pn_RB[Bin] = len([yRanked for i in range(N) #count number of point between y bounds of bin
            if yRanked[i] >= PC[Bin].get('nodes')[2] and yRanked[i] <= PC[Bin].get('nodes')[3]])
    Pn_AR = (Pn_AR)/N #divide for probability
    Pn_RB = (Pn_RB)/N
    
    # find the approximation for the mutual information function
    Iapprox = np.dot(Pn_AB, np.log(Pn_AB/(Pn_AR*Pn_RB)))
    return Iapprox

def MI_delay(time_series):
    
    """This function returns a suitable embedding delay, given a time series based on the mutual information algorithm.

    Args:
       ts (array):  Time series array (1d).

    Returns:
       (int): tau, The embedding delay for delay embedding reconstruction.

    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from scipy import sparse
    from ripser import ripser
    
    #_sweep through various delays_
    delayMax = 100 #maximum delay set by user
    I = np.zeros(delayMax) #initializes information array
    I[0] = 10**10
    end = False
    delay = 0
    tau = 0
    step = 2
    i = 0
    while end == False:
        i = i+1
        delay = delay + step
        x  = ts[:-delay] #takes all terms from time series besides last (delay) terms
        y = ts[delay:] #takes all terms from time series besides first (delay) terms
        I[i] = mutualInfo(x, y) #calculates mutual information\
        # __Plotting___
        if I[i]>I[i-1]:
            tau = delay-1
            end = True
                
        if delay > delayMax:
            end = True
    
    return tau

def delay_op(time_series, plotting = False):
    
    """This function returns a suitable embedding delay, given a time series, based on the multi-scale permutation entropy algorithm.

    Args:
       ts (array):  Time series array (1d).

    Kwargs:
       plotting (bool): If True then the function will also plot the multi-scale permutation entropy plot for validation purposes.

    Returns:
       (int): tau, The embedding delay for permutation formation.

    """
    
    #returns delay based on equioprobable permutations
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from scipy import sparse
    from ripser import ripser
    from pyentrp import entropy as ent
    import math
    import numpy as np
    
    flag = False
    trip = 0.975 #sets trip value for knowing there is approximately a peak
    MSPE = []
    delays = []
    ds = 1 #delay start
    de = 200 #delay end
    m = 3 #embedding dimension
    start = False #hasn't started yet
    end = False #hasn't found the first peak after noise yet
    delay = ds
    NPE_previous = 0 #initialize normalized permutation entropy (NPE)
    while end == False: #While it hasn't found a peak
        ME = np.log2(math.factorial(m))
        PE = ent.permutation_entropy(time_series,m,delay)
        NPE = PE/ME
        delays.append(delay)
        if NPE < trip:
            start = True
        if NPE > trip and start == True and end == False:
            delay_peak = delay
            end = True
            NPE_previous = NPE
        MSPE = np.append(MSPE, NPE)
        if delay > de: #if it reaches the end
            if flag == True: #if it didn't find an appropriate delay after the second pass
                print('Warnings (1), (2), or (3):')
                print('(1) Amplitude of noise is too large and exceeds signal.')
                print('(2) If no noise or small amount, then time series is signficantly over sampled with')
                print('    Sampling Frequency greater than 100 times Nyquist rate.')
                      
                print('(3) The time series is from a discrete signal (Map).')
                print('*** Delay = 1 is returned ***')
                delay_peak = 1 #return delay of 1 (probably this is a map)
                end = True
            trip = trip + 0.025 #try again but higher trip value
            flag = True #sets warning flag that this is the second pass 
            delay = 1 #restartes delay to 1
        delay = delay+1
        
    L = len(time_series)
    if L/delay_peak < 50:
        print('Warning: delay may be inaccurate due to short time series')
    if plotting == True:
        plt.figure(figsize=(5,3))
        plt.plot(delays, MSPE,'k.')
        plt.xlabel(delay)
        plt.ylabel('NPE')
        plt.show()
    return delay_peak


def n_op(time_series, delay):
    """This function returns a suitable embedding dimension, given a time series, based on the multi-scale permutation entropy algorithm.

    Args:
       delay (int): dimension n from the method of multi-scale permutation entropy.
       ts (array):  Time series array (1d).

    Returns:
       (int): n, The embedding dimension for permutation formation.

    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from scipy import sparse
    from ripser import ripser
    from pyentrp import entropy as ent
    import numpy as np
    
    #returns an embedding dimenion suitable based on maximum permutation entropy per symbol
    m_start, m_end = 3, 7
    
    MnPE = []
    for m in range(m_start,m_end+1):
        PE = ent.permutation_entropy(time_series,m,delay)/(np.log(2))
        NPE = PE/(m-1)
        MnPE = np.append(MnPE, NPE)
    dim = np.argmax(MnPE)
    return dim+m_start


def Permutation_Sequence(time_series, m, delay): #finds permutation sequency from modified pyentropy package
    
    """Given the time series and the assocaited permutation formation dimention n and tau, this function will generate a sequence of time-ordered permutations.
    
    Args:
       delay (int): tau from the method of multi-scale permutation entropy.
       n (int): n from the method of multi-scale permutation entropy.
       ts (array):  Time series array (1d).

    Returns:
       (array): S, The sequence of time-ordered permutations

    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from scipy import sparse
    from ripser import ripser
    import itertools
    import numpy as np
    #inputs: time_series = time series (1-d), m = embedding or motif dimension, and delay = embedding delay from function delay_op
    
    def util_hash_term(perm): #finds permutation type
        deg = len(perm)
        return sum([perm[k]*deg**k for k in range(deg)])
    L = len(time_series) #total length of time series
    perm_order = [] #prepares permutation sequence array
    permutations = np.array(list(itertools.permutations(range(m)))) #prepares all possible permutations for comparison
    hashlist = [util_hash_term(perm) for perm in permutations] #prepares hashlist
    for i in range(L - delay * (m - 1)): 
    #For all possible permutations in time series
        sorted_index_array = np.array(np.argsort(time_series[i:i + delay * m:delay], kind='quicksort')) 
        #sort array for catagorization
        hashvalue = util_hash_term(sorted_index_array);
        #permutation type
        perm_order = np.append(perm_order, np.argwhere(hashlist == hashvalue)[0][0])
        #appends new permutation to end of array
    perm_seq = perm_order.astype(int)+1 #sets permutation type as integer where $p_i \in \mathbb{z}_{>0}$
    return perm_seq #returns sequence of permutations


def AdjacenyMatrix_OP(perm_seq, n): #Gets Adjacency Matrix (weighted and direction) using permutation sequence
    """Using a permutation sequence, this function will generate an directed and weighted adjacency matrix counting the number of transitions between permutaton types.
   
    Args:
       n (int): dimension n from the method of multi-scale permutation entropy.
       S (array):  Permutation sequence.

    Returns:
       (array): A, An NxN weighted and directed arjacency matrix with N permutation types.
       
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from scipy import sparse
    from ripser import ripser
    import numpy as np
    
    N = np.math.factorial(n) #number of possible nodes
    A = np.zeros((N,N)) #prepares A
    for i in range(len(perm_seq)): #go through all permutation transitions (This could be faster wiuthout for loop)
        A[perm_seq[i-1]-1][perm_seq[i]-1] += 1 #for each transition between permutations increment A_ij
    A = A[~(A==0).all(1)] #removes all columns with all elements == 0 (node desn't exist)
    A = A[:,~(A==0).all(0)] #removes all rows with all elements == 0 (node desn't exist)
    return A #this A is directional and weighted



def DistanceMatrix_OP(A, weighted = False, shortest_path = True):
    """This function uses the adjacency matrix to form a distance matrix. The distance matrix be 
    interpretted from an unweighted or weighted version of the adjacenecy matrix A, and the shortest path does not need to be used.
    However, our work suggests using an unweighted adjacency matrix with a shortest path distance metric.
    
    Args:
       A (array): An NxN weighted and directed arjacency matrix with N permutation types.

    Kwargs:
       weighted (bool): If True then the function will use a weighted version of the adjacency matrix.
       shortest_path (bool): If True then the function will use the shortest distance path length between nodes. 
       
    Returns:
       (array): D, An NxN undirected distance matrix.

    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from scipy import sparse
    from ripser import ripser
    import networkx as nx
    
    #inputs: A = weighted, directional adjacency matrix, weighted = weighting on A for calculating distance matrix 
    
    if weighted == True:
        A = A + A.T #make undirected adjacency matrix
        np.fill_diagonal(A, 0) #get rid of diagonal for proper weighting (diagonal can have very high values from self transitions)
        Amax = np.nanmax(A) #finds max of A for transformation into distance matrix
        Dmat = Amax - A + 1 # flips distance matrix so that high edge weights = short distance
        Dmat[Dmat==np.nanmax(Dmat)] = np.inf  #replaces max distance with infinity because this would represent no connection
        np.fill_diagonal(Dmat, np.inf) #gets rid of diagonal byu setting distance to infinity
    if weighted == False: #if unweighted
        A = A + A.T #make undirected adjacency matrix
        A[A>0] = 1 #sets all edge weights = 1
        if shortest_path == True: #leave Distance matrix as adjacency if shortest path will be applied 
            Dmat = A 
            np.fill_diagonal(Dmat, 0) #get rid of diagonal as we don't care about self transitions
        if shortest_path == False: #otherwise appropriate apply distances
            A[A==0] = np.inf
            Dmat = A #if direct connection doesn't exist, make infinite distance
            np.fill_diagonal(Dmat, np.inf) #get rid of diagonal as we don't care about self transitions
        
    
    if shortest_path == True:
        G = nx.Graph()
        s_arr = [] #array of starts
        e_arr = [] #array of ends
        w_arr = [] #array of edge weights
        for s in range(len(A[0])): #finds all connections between nodes
            for e in range(len(A[0])):
                if np.isinf(Dmat[s][e]) == False and Dmat[s][e] != 0: #if a path exists
                    s_arr = np.append(s_arr, s)
                    e_arr = np.append(e_arr, e) 
                    w_arr = np.append(w_arr, Dmat[s,e]) 
                    
        s_arr =  s_arr.astype(int) #start array
        e_arr =  e_arr.astype(int) #end array
        w_arr =  w_arr.astype(int) #weight array
        
        if weighted == False:
            edges = zip(s_arr, e_arr) # zips edges together
            G.add_edges_from(edges) #adds all edges into graph G
            length = dict(nx.all_pairs_shortest_path_length(G)) #finds shortest distance between edges
            
        if weighted == True:
            edges_w = zip(s_arr, e_arr, w_arr) # zips edges and weights together
            G.add_weighted_edges_from(edges_w) #adds all weighted edges into graph G
            length = dict(nx.all_pairs_dijkstra_path_length(G)) #finds shortest distance between edges
        
        for s in range(len(A[0])):
            for e in range(len(A[0])):
                if Dmat[s][e] ==  0 or np.isinf(Dmat[s][e]) == True: #if a direct connection doesn't exist
                    Dmat[s][e] = length[s][e]
    
    return Dmat

    
    

def Takens_Embedding(ts, n, tau):
    """This function returns a suitable embedding delay, given a time series, based on the multi-scale permutation entropy algorithm.

    Args:
       delay (int): tau from the method of multi-scale permutation entropy.
       ts (array):  Time series array (1d).

    Returns:
       (int): n, The embedding dimension for permutation formation.

    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from scipy import sparse
    from ripser import ripser
    
    #takens embedding method. Not the fastest algoriothm, but it works. Need to improve
    ets = np.tile(ts,(len(ts),1))
    li = np.tril_indices(len(ts), k = -1)
    ui = np.triu_indices(len(ts), k = tau*(n-1)+1)
    ets[li] = 0
    ets[ui] = 0
    ets = ets[:-(tau*(n-1))]
    a = []
    for i in range(len(ets)):
        b = np.trim_zeros(ets[i])
        a = np.append(a, b[::tau])
    if len(a)%n != 0:
        a = a[len(a)%n:]
        ets = a.reshape(len(ets)-1,n)
    
    else:
        ets = a.reshape(len(ets),n)
    return ets

def k_NN(embedded_time_series, k):
    """This function returns a suitable embedding delay, given a time series, based on the multi-scale permutation entropy algorithm.

    Args:
       delay (int): tau from the method of multi-scale permutation entropy.
       ts (array):  Time series array (1d).

    Returns:
       (int): n, The embedding dimension for permutation formation.

    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from scipy import sparse
    from ripser import ripser
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(emb_ts) #get nearest neighbors
    neighbordistances, indices = nbrs.kneighbors(emb_ts) #get incidices of nearest neighbors
    from scipy.spatial import distance
    distances = distance.cdist(emb_ts, emb_ts, 'euclidean') #get euclidean distance between embedded vectors
    return distances, indices
    
    
def Adjacency_KNN(indices):
    """Uses the nearest neighbor indices to form adjacency matrix (weighted and undirected).
    
    Args:
       I (array): an array of indices.

    Returns:
       (array): A, An NxN unweighted and undirected adjacency matrix. The output is unweighted since a self 
       repeated nearest neighbor is the only chance for a weighting to be applied.
       
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from scipy import sparse
    from ripser import ripser
    
    A = np.zeros((len(indices.T[0]),len(indices.T[0])))
    for h in range(len(indices.T[0])):
        KNN_i = indices[h][indices[h]!=h] #indices of k nearest neighbors
        A[h][KNN_i]+=1 #increment A_ij for kNN indices
        A.T[h][KNN_i]+=1
    A[A>0] = 1
    return A


def DistMatrix_KNN(A, distances, weighted, shortest_path):
    """This function uses the adjacency matrix to form a distance matrix. The distance matrix can be 
    interpretted from an unweighted or weighted version of the adjacenecy matrix A, and the shortest path does not need to be used.
    However, our work suggests using an unweighted adjacency matrix with a shortest path distance metric.
    
    Args:
       A (array): An NxN weighted and directed arjacency matrix with N permutation types.

    Kwargs:
       weighted (bool): If True then the function will use a weighted version of the adjacency matrix.
       shortest_path (bool): If True then the function will use the shortest distance path length between nodes. 
       
    Returns:
       (array): D, An NxN undirected distance matrix, where N is the number of embedded vectors.
       

    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from scipy import sparse
    from ripser import ripser
    import networkx as nx
    
    if weighted == True:
        Dmat = np.copy(A) 
        Dmat[Dmat==0] = np.inf #set disconnected nodes to infinity
        Dmat[Dmat==1] = distances[Dmat==1] #set connected nodes to the euclidean distance between nodes
        
    if weighted == False:
        Dmat = np.copy(A)
        Dmat[Dmat==0] = np.inf #set disconnected nodes to infinity
    ERROR = False #flag for checking if embedding multiplicity error occurs
    
    if shortest_path == True:
        
        G = nx.Graph() #initialize graph
        s_arr = [] #array of starts
        e_arr = [] #array of ends
        w_arr = [] #array of edge weights
        for s in range(len(A[0])): #finds all connections between nodes
            for e in range(len(A[0])):
                if np.isinf(Dmat[s][e]) == False: #if a direct connection exists
                    s_arr = np.append(s_arr, s)
                    e_arr = np.append(e_arr, e) 
                    w_arr = np.append(w_arr, Dmat[s,e]) 
                    
        s_arr =  s_arr.astype(int) #start array
        e_arr =  e_arr.astype(int) #end array
        
        if weighted == False:
            edges = zip(s_arr, e_arr) # zips edges together
            G.add_edges_from(edges) #adds all edges into graph G
            length = dict(nx.all_pairs_shortest_path_length(G)) #finds shortest path between nodes
            
        if weighted == True:
            edges_w = zip(s_arr, e_arr, w_arr) # zips edges and weights together
            G.add_weighted_edges_from(edges_w) #adds all weighted edges into graph G
            length = dict(nx.all_pairs_dijkstra_path_length(G)) #finds shortest distance between nodes
        
        for s in range(len(A[0])):
            for e in range(len(A[0])):
                if np.isinf(Dmat[s][e]) == True: #if a direct connection doesn't exist
                    try:
                        Dmat[s][e] = length[s][e] #if a path exists
                    except KeyError:
                        Dmat[s][e] = np.inf #if no path exists then set as infinitely far away
                        ERROR = True #flag embedding multipcity error
    if ERROR == True:
        print('Error: Embedding multiplicity causing disconnecting network')
        
    return Dmat


def MakeNetwork(A):
    """This function takes the adjacency matrix and forms a list of node positions and 
    edges based on the unweighted and undirected version of the network.
    
    Args:
       A (array): An NxN weighted and directed arjacency matrix.

    Returns:
       (array): pos, graph and position of nodes based on an electric-spring layout.

    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from scipy import sparse
    from ripser import ripser
    import networkx as nx
    
    #ordinal network stuff
    A = A + A.T #make undirected adjacency matrix
    np.fill_diagonal(A, 0) #get rid of diagonal 
    A[A>0] = 1 #make unweighted
    
    G = nx.Graph()
    G.add_nodes_from(range(len(A[0])))
        
    edges1 = []
    edges2 = []
    for h in range(0,len(A[0])):
        edges1 = np.append(edges1, np.nonzero(A[h]))
        L = len(np.nonzero(A[h])[0])
        edges2 = np.append(edges2, np.zeros(L) + h)
    edges1 = edges1.astype(int)+1
    edges2 = edges2.astype(int)+1
    edges = zip(edges1,edges2)
    G.add_edges_from(edges)
    if 0 in G.nodes():
        G.remove_node(0)
        
    pos = nx.spring_layout(G, iterations = 2000)
    
    return G, pos
    
  
# In[ ]:
    
    
if __name__ == "__main__": #Only runs if running from this file (This will show basic example)
    
    #import needed packages
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx
    from scipy import sparse
    from ripser import ripser

    t = np.linspace(0,40,400)
    ts = np.sin(t)
    
    
    op = True
    if op == True:
        #example process for ordinal partition network
        print('ORDINAL PARTITIONS METHOD')
        tau = delay_op(ts)
        n = 6
        Sample = tau*100
        ts = (ts)[0:Sample]
        t = (t)[0:Sample]
        #delay from embedding lag causing equiprobable permutations
        
        #dimension from motif dimension with the highest permutation entropy per symbol
        #However, n = 6 usually provides the best results.
        print('delay:     ', tau)
        print('dimension: ', n)
        
        PS = Permutation_Sequence(ts,n,tau)
        #Gets a sequence of permutations from time series
        
        A = AdjacenyMatrix_OP(PS, n)
        #gets adjacency matrix from permutation sequence transtitions
    
        D = DistanceMatrix_OP(A, weighted = False, shortest_path = True)   
        #gets distance matrix from adjacency matrix with weighting as option and shortest path as option.
    
        G, pos = MakeNetwork(A)
        #makes graph from adjacency (unweighted, non-directional) matrix
        
        D_sparse = sparse.coo_matrix(D).tocsr()
        result = ripser(D_sparse, distance_matrix=True, maxdim=1)
        diagram = result['dgms']
        
        plotting = True    
        if plotting == True:
            
            TextSize = 12
            MS = 4
            plt.figure(1) 
            plt.figure(figsize=(6,9))
            gs = gridspec.GridSpec(3, 2) 
                
            ax = plt.subplot(gs[0, 0]) #plot time series
            plt.title('Time Series', size = TextSize)
            plt.plot(t, ts)
            plt.xticks(size = TextSize)
            plt.yticks(size = TextSize)
            
            ax = plt.subplot(gs[0, 1]) #plot time series
            plt.title('Permutation Sequence', size = TextSize)
            plt.plot(PS)
            plt.xticks(size = TextSize)
            plt.yticks(size = TextSize)
            
            ax = plt.subplot(gs[1, 0]) #plot time series
            plt.title('Adjacency Matrix', size = TextSize)
            plt.imshow(A)
            plt.colorbar()
            plt.xticks(size = TextSize)
            plt.yticks(size = TextSize)
    
            ax = plt.subplot(gs[2, 0]) #plot time series
            plt.title('Distance Matrix', size = TextSize)
            plt.imshow(D)
            plt.colorbar()
            plt.xticks(size = TextSize)
            plt.yticks(size = TextSize)
              
            
            ax = plt.subplot(gs[1, 1]) #plot time series
            plt.title('Network', size = TextSize)
            nx.draw(G, pos, with_labels=False, font_weight='bold', node_color='blue',
                    width=1, font_size = 10, node_size = 20)
            
            ax = plt.subplot(gs[2, 1]) #plot time series
            plt.title('Persistence Diagram', size = TextSize)
            plt.xticks(size = TextSize)
            plt.yticks(size = TextSize)
            plt.plot(diagram[0].T[0], diagram[0].T[1], 'ro')
            plt.plot(diagram[1].T[0], diagram[1].T[1], 'bs')
            plt.plot([0, max(diagram[1].T[1])], [0, max(diagram[1].T[1])], 'k--')
            
            
            plt.subplots_adjust(hspace= 0.5)
            plt.subplots_adjust(wspace= 0.5)
            plt.savefig('C:\\Users\\myersau3.EGR\\Desktop\\python_png\\networks_example_OP.png', bbox_inches='tight',dpi = 400)
            plt.show()
    
    
    knn = True
    if knn == True:
        print('K NEAREST NEIGHBORS METHOD')
        #example process for k nearest neighbors network from Takens' embedding
        
        tau = MI_delay(ts) 
        #Mutual information isn't producing an accurate delay. Need to fix.
        DownSample = tau/4 #downsampling to allow for longer time series
        tau = int(tau/DownSample) 
        print('delay:     ', tau)
        n = FNN_dim(tau,ts)+1 
        #embedding dimension from FNN +1 to make sure its high enough dimension
        print('dimension: ', n)
        
        sample = 300
        t = t[::int(DownSample)][:sample]
        ts = ts[::int(DownSample)][:sample]
        
        
        emb_ts = Takens_Embedding(ts, n, tau) 
        #takens embedding of time series in n dimenional space with delay tau
        
        distances, indices = k_NN(emb_ts, k= 4) 
        #gets distances between embedded vectors and the indices of the nearest neighbors for every vector
        
        A = Adjacency_KNN(indices)
        #get adjacency matrix (weighted, directional)
        
        G, pos = MakeNetwork(A)
        #get network graph based on adjacency matrix (unweighted, non-directional)
              
        D = DistMatrix_KNN(A, distances, weighted = False, shortest_path = True)
        #get distance matrix. Specify if weighting is desired or shortest path 
          
        D_sparse = sparse.coo_matrix(D).tocsr()
        result = ripser(D_sparse, distance_matrix=True, maxdim=1)
        diagram = result['dgms']
    
        plotting = True    
        if plotting == True:
            TextSize = 12
            MS = 4
            plt.figure(2) 
            plt.figure(figsize=(6,9))
            gs = gridspec.GridSpec(3, 2) 
                
                
            ax = plt.subplot(gs[0, 0]) #plot time series
            plt.title('Time Series', size = TextSize)
            plt.plot(t,ts)
            plt.xticks(size = TextSize)
            plt.yticks(size = TextSize)
            
            ax = plt.subplot(gs[0, 1]) #plot time series
            plt.title('Takens Embedded (2D)', size = TextSize)
            plt.plot(emb_ts.T[0],emb_ts.T[1])
            plt.xticks(size = TextSize)
            plt.yticks(size = TextSize)
            
            ax = plt.subplot(gs[1, 0]) #plot time series
            plt.title('Adjacency Matrix', size = TextSize)
            plt.imshow(A)
            plt.colorbar()
            plt.xticks(size = TextSize)
            plt.yticks(size = TextSize)
    
            ax = plt.subplot(gs[2, 0]) #plot time series
            plt.title('Distance Matrix', size = TextSize)
            plt.imshow(D)
            plt.colorbar()
            plt.xticks(size = TextSize)
            plt.yticks(size = TextSize)
              
            
            ax = plt.subplot(gs[1, 1]) #plot time series
            plt.title('Network', size = TextSize)
            nx.draw(G, pos, with_labels=False, font_weight='bold', node_color='blue',
                    width=1, font_size = 10, node_size = 20)
            
            
            ax = plt.subplot(gs[2, 1]) #plot time series
            plt.title('Persistence Diagram', size = TextSize)
            plt.xticks(size = TextSize)
            plt.yticks(size = TextSize)
            plt.plot(diagram[0].T[0], diagram[0].T[1], 'ro')
            plt.plot(diagram[1].T[0], diagram[1].T[1], 'bs')
            if len(diagram[1].T[1]) > 0:
                plt.plot([0, max(diagram[1].T[1])], [0, max(diagram[1].T[1])], 'k--')
            
            
            plt.subplots_adjust(hspace= 0.5)
            plt.subplots_adjust(wspace= 0.5)
            plt.savefig('C:\\Users\\myersau3.EGR\\Desktop\\python_png\\networks_example_kNN.png', bbox_inches='tight',dpi = 400)
            plt.show()
            
    
            
          
          
    