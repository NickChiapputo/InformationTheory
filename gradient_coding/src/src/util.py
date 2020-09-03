from __future__ import print_function
import sys
import random
import os
import numpy as np
import itertools
import scipy.special as sp
from scipy.sparse import csr_matrix

# ---- Data generation, saving, loading and modification routines

def load_data(input_file):
    mydata = np.loadtxt(input_file,dtype=float)
    return mydata
    
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename+".npz")
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def save_matrix(m, output):
    f = open(output, "w")
    for i in range(len(m)):
        print(" ".join([str(x) for x in m[i]]), file=f)
    f.close()

def save_vector(m, output):
    f = open(output, "w")
    for i in range(len(m)):
        print("%5.3f" %(m[i])+" ", file=f)
    f.close()

# generates a random matrix representing samples from a two-component GMM with identity covariance
def generate_random_matrix_normal(mu1, mu2, n_rows, n_cols):
    ctrmu2 = np.random.binomial(n_rows,0.5)
    ctrmu1 = n_rows - ctrmu2 
    mfac = 10/np.sqrt(n_cols)
    return np.concatenate((np.add(mfac*np.random.standard_normal((ctrmu1, n_cols)), mu1), np.add(mfac*np.random.standard_normal((ctrmu2, n_cols)), mu2)))

# generates a vector of random labels, each entry only has value -1 or 1
def generate_random_binvec(n):
    return np.array([np.random.randint(2)*2-1 for x in range(n)])

def interactionTermsAmazon(data, degree, hash=hash):
    # Create empty list to hold data.
    new_data = []

    # Get the number of rows $m$ and number of columns $n$.
    m,n = data.shape

    # Iterate through each degree-tuple of terms in the columns.
    # Order matters for the tuples. That is, all tuples (i, j) where 0 <= i < j < n will exist.
    # For example, given n = 3 columns and degree = 2, get all 2-tuples of [ 0, 1, 2 ]:
    #       ( 0, 1 ), ( 0, 2 ), ( 1, 2 )
    # After the for loop is complete, new_data will contain $n$ rows that each contain a list of hash values of size $m$.
    print(  "================================================================\n" \
            "                           " + "\033[1;31m" + "IterTools\n" + "\033[0;0m" \
            "n = {}\ndegree = {}\nindices = ".format( n, degree ), end = '' )
    count = 0
    for indicies in itertools.combinations(range(n), degree):

        # Display the tuples.
        print( "{}, ".format( indicies ), end = '' )
        count = count + 1
        if( count == 5 ):
            print( "\n          ", end = '' )
            count = 0

        # Add the tuple to the list if it is not (5, 7) or (2, 3).
        # (5, 7) = ROLE_TITLE    and ROLE_FAMILY
        # (2, 3) = ROLE_ROLLUP_1 and ROLE_ROLLUP_2
        if not(5 in indicies and 7 in indicies) and not(2 in indicies and 3 in indicies):
            # data[ :, indicies ]   -> Return the tuple of values at the indicies provided for each row of the data
            # hash( tuple( v ) )    -> Create a tuple of the values from the data and hash it to a single value.
            # Combined, this creates a list of length equal to the number of rows in data.
            new_data.append([hash(tuple(v)) for v in data[:, indicies]])

    # Count the number of tuples selected.
    print( "\nNumber of Data Points = {}".format( len( new_data ) ) )

    print(  "Transposed Data = {}\n" \
            "      Data Size = {}\n" \
            "================================================================\n".format( ( np.array( new_data ).T ), ( np.array( new_data ).T ).shape ) )

    # Return the transpose of the new data.
    # Size of data is m x n (just like the original data).
    return np.array(new_data).T

# ---- Other routines 

def unique_rows(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    return a[idx]

def getB(n_workers,n_stragglers):
    Htemp=np.random.normal(0,1,[n_stragglers,n_workers-1])
    H=np.vstack([Htemp.T,-np.sum(Htemp,axis=1)]).T

    Ssets=np.zeros([n_workers,n_stragglers+1])

    for i in range(n_workers):
        Ssets[i,:]=np.arange(i,i+n_stragglers+1)
    Ssets=Ssets.astype(int)
    Ssets=Ssets%n_workers
    B=np.zeros([n_workers,n_workers])
    for i in range(n_workers):
        B[i,Ssets[i,0]]=1
        vtemp=-np.linalg.solve(H[:,np.array(Ssets[i,1:])],H[:,Ssets[i,0]])
        ctr=0
        for j in Ssets[i,1:]:
            B[i,j]=vtemp[ctr]
            ctr+=1

    return B

def getA(B,n_workers,n_stragglers):
    #S=np.array(list(itertools.permutations(np.hstack([np.zeros(n_stragglers),np.ones(n_workers-n_stragglers)]),n_workers)))
    #print(S)
    #S=unique_rows(S)
    
    S = np.ones((int(sp.binom(n_workers,n_stragglers)),n_workers))
    combs = itertools.combinations(range(n_workers), n_stragglers)
    i=0
    for pos in combs:
        S[i,pos] = 0
        i += 1

    (m,n)=S.shape
    A=np.zeros([m,n])
    for i in range(m):
        sp_pos=S[i,:]==1
        A[i,sp_pos]=np.linalg.lstsq(B[sp_pos,:].T,np.ones(n_workers))[0]

    return A

def compare(a,b):
    for id in range(len(a)):
        if a[id] and (not b[id]):
            return 1
        if (not a[id]) and b[id]:
            return -1
    return 0

def binary_search_row_wise(Aindex,completed,st,nd):
    if st>=nd-1:
        return st
    idx=(st+nd)/2
    cp=compare(Aindex[idx,:],completed)
    if (cp==0):
        return idx
    elif (cp==1):
        return binary_search_row_wise(Aindex,completed,st,idx)
    else:
        return binary_search_row_wise(Aindex,completed,idx+1,nd)

def calculate_indexA(boolvec):
    l = len(boolvec)
    ctr = 0
    ind = 0
    for j in range(l-1,-1, -1):
        if boolvec[j]:
            ctr = ctr+1
            ind = ind + sp.binom(l-1-j, ctr)

    return int(ind)

def calculate_loss(y,predy,n_samples):
    return np.sum(np.log(1+np.exp(-np.multiply(y,predy))))/n_samples