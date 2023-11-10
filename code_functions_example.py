# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:55:13 2023

@author: inesmpsoares
"""

#%% CASE STUDY 1 - HC

#Process Data
import numpy as np
import pandas as pd

#Read the CSV file into a DataFrame

def read_data(filepath):
    """
    Reads a CSV file from the specified filepath, performs data manipulation, and returns a pandas DataFrame.
    Parameters:
        filepath (str): The path to the CSV file to be read.
    Returns:
        dataframe (pandas DataFrame): A DataFrame containing the data from the CSV file after manipulation.
   """
    dataframe = pd.read_csv(filepath, delimiter=",", header=0, index_col=0)
    #dataframe.drop(dataframe.columns[-3:], axis=1, inplace=True)
    return dataframe

filepath = 'C:/Users/inesm/Documents/Tese/Datasets/logCPM_centered.csv'
case_study1 = pd.read_csv(filepath, delimiter=",", header=0, index_col=0)



# Define what is going to be clustered
def what_to_cluster(data_frame, cluster_by):
    
    """
    Define the data frame to use for clustering based on the cluster_by parameter.

    Parameters:
        data_frame: A pandas DataFrame object containing the data to be clustered.
        cluster_by: A string specifying whether to cluster by 'gene' or 'sample'.

    Returns:
        data_cluster: A pandas DataFrame object representing the data frame to be used for clustering.
    """
    if cluster_by == 'gene':
        data_cluster = data_frame
        # If cluster_by is 'gene', the input data_frame is assigned directly to the data_cluster variable.
        
    if cluster_by == 'sample':
        data_cluster = data_frame.transpose(copy=True)
        # If cluster_by is 'sample', the data_frame is transposed (rows become columns) and the resulting transposed data frame is assigned to the data_cluster variable.

    else:
        print ('error = cluster_by must be gene or sample')
        # If cluster_by has any other value, an error message is printed.
    
    return data_cluster


#Hierarchical Clustering
import scipy
import scipy.cluster.hierarchy as sch
import pandas as pd
from scipy.cluster.hierarchy import _LINKAGE_METHODS, dendrogram, linkage, set_link_color_palette
import fastcluster as fc


# Hierarchical Clustering with a desired method and metric
def hierarchical_matrix(data_frame, method, metric):
    """
    Performs hierarchical clustering on a given data frame using the specified parameters.
    
    Args:
        data_frame (pandas.DataFrame): The input data frame containing the data to be clustered.
        method: A string parameter specifying the linkage method to be used for clustering. 
                The method options are 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'
        metric: A string parameter specifying the distance metric to be used for clustering.
                The metric options are 'euclidean', 'cosine', 'correlation'
    Returns:
        The resulting hierarchical clustering linkage matrix.
    """
    
    #preserve_input= false, usa menos memoria
    
    linkage_matrix = fc.linkage(data_frame, method=method, metric=metric, preserve_input='True')
            
    return linkage_matrix

CS1=what_to_cluster(case_study1, 'sample')
HC_CS1= hierarchical_matrix(CS1, 'complete', 'correlation') #linkage_matrix


import biosppy
from biosppy.clustering import _life_time
from scipy.cluster.hierarchy import fcluster

def hierarchical_partition(linkage_matrix, criterion, n):
    """
    Performs hierarchical partitioning on a given linkage matrix.

    Parameters:
        linkage_matrix (ndarray): The linkage matrix obtained from hierarchical clustering.
        criterion (str): The criterion for partitioning. It can be either 'maxcluster' or 'life_time'.
        n (float): Number of clusters to form or life time if 0.

    Returns:
        threshold: Float that corresponds to the threshold to form n number of clusters.
        labels: An array of labels indicating the cluster assignments for each sample.
    """
    if  criterion == 'maxcluster':
        th=((linkage_matrix[-n,2]+linkage_matrix[-(n-1),2])/2)
        labels = fcluster(linkage_matrix, n, criterion='maxclust')
        
    if  criterion == 'life_time':
        N = (len(linkage_matrix)+1)

        if N < 3:
            raise ValueError("The number of objects N must be greater then 2.")

        # compute differences from Z distances
        df = np.diff(linkage_matrix[:, 2])
        # find maximum difference
        idx_max = np.argmax(df)
        mx_dif = df[idx_max]
        # find minimum difference
        mi_dif = np.min(df[np.nonzero(df != 0)])

            
        # find threshold link distance
        th_link = linkage_matrix[idx_max, 2]
        # links above threshold
        idxs = linkage_matrix[np.nonzero(linkage_matrix[:, 2] > th_link)[0], 2]
        #number of links above threshold +1 = number of clusters and singletons
        cont = len(idxs) + 1

        # condition (perceber melhor)
        if mi_dif != mx_dif:
            if mx_dif < 2 * mi_dif:
                cont = 1

        if cont > 1:
            labels = fcluster(linkage_matrix, cont, 'maxclust')
        else:
            labels = np.arange(N, dtype='int')
 
        
        th = ((linkage_matrix[idx_max, 2]+linkage_matrix[idx_max+1,2])/2)
    
    return {'threshold':th, 'samples_labels':labels}

HC_partition_CS1 = hierarchical_partition(HC_CS1, 'life_time',0)


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sb
from seaborn import color_palette


def plot_dendrogram(linkage_matrix, sample_names, cluster_threshold, cmap): # plot the hierarchical clustering
    """
   Plot a Dendrogram
    
    This function generates and displays a dendrogram plot based on the provided linkage matrix,
    which represents the hierarchical clustering of data points. The dendrogram illustrates the
    hierarchical structure of clusters in the data, with vertical lines indicating cluster
    mergers at different levels of similarity.
    
    Parameters:
    - linkage_matrix (array-like): The linkage matrix resulting from hierarchical clustering,
      defining how clusters are merged.
    - labels (list or array-like): Labels or identifiers for the data points being clustered.
    - cluster_threshold (float): A threshold value to color clusters above it differently,
      aiding in the identification of meaningful clusters.
    - cmap (str or colormap, optional): The colormap used for coloring clusters.
    
    Returns:
    - None: The function displays the dendrogram plot but does not return any values.
    
    """
    plt.figure(figsize=(20,8), dpi=300)
    plt.title("Average Linkage", fontsize=20)
    plt.xlabel("Samples")
    plt.ylabel("Cosine Distance", fontsize=20)
    sbcmap = sb.color_palette()
    set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in sbcmap])
    dendrogram(Z=linkage_matrix, color_threshold=cluster_threshold, labels=sample_names, above_threshold_color='#b3b3b3', leaf_font_size=12) #codigo do cinzento
    
    return plt.show()


sbcmap = sb.color_palette("RdYlBu")
dendogram_CS1 = plot_dendrogram(HC_CS1, CS1.index, HC_partition_CS1['threshold'], sbcmap)



from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

def evaluate_clustering(data_frame, labels): # Evaluate the performed clusters
    """
    Evaluate the quality of hierarchical clustering results using multiple evaluation metrics.

    Parameters:
        data_frame: A pandas DataFrame object containing the data used for clustering.
        linkage_matrix: The resulting hierarchical clustering linkage matrix.

    Returns:
        A dictionary containing the evaluation scores:
        - 'calinski_harabasz': The Calinski-Harabasz Index.
        - 'davies_bouldin': The Davies-Bouldin Index.
        - 'silhouette': The Silhouette Coefficient.
    """

    # Calculate evaluation scores
    scores = {}

    # Calinski-Harabasz Index
    ch_score = calinski_harabasz_score(data_frame, labels)
    scores['calinski_harabasz'] = ch_score

    # Davies-Bouldin Index
    db_score = davies_bouldin_score(data_frame, labels)
    scores['davies_bouldin'] = db_score

    # Silhouette Coefficient
    silhouette_avg = silhouette_score(data_frame, labels)
    scores['silhouette'] = silhouette_avg

    return scores


scores_CS1 = evaluate_clustering(CS1, HC_partition_CS1['samples_labels'] )

#%% CASE STUDY 1- Bi-clustering
import seaborn as sb

def bicluster(data, method, metric, figsize=(20, 20), dendrogram_ratio=(0.2, 0.2), cmap='Spectral_r'):
    """
    Performs biclustering on the given data using the specified parameters and visualizes the results.

    Parameters:
        data (pandas.DataFrame or numpy.ndarray): The input data to be biclustered.
        method (str): The linkage method to be used for clustering. Available options are:
                      - 'single', complete','average','weighted','centroid','median','ward'.
        metric (str): The distance metric to be used for clustering. Available options are:
                      - 'euclidean','cosine','correlation'
        figsize (tuple, optional): The figure size for the resulting clustermap. Defaults to (20, 20).
        dendrogram_ratio (tuple, optional): The ratio of the dendrogram sizes. Defaults to (0.2, 0.2).
        cmap (str or colormap, optional): The colormap to be used for the resulting clustermap. Defaults to 'Spectral_r'.

    Returns:
        seaborn.matrix.ClusterGrid: The resulting clustermap object.
    """
    sb.set_theme(color_codes=True)
    bicluster_grid = sb.clustermap(data, method=method, metric=metric, 
                                   figsize=figsize, dendrogram_ratio=dendrogram_ratio, cmap=cmap) #col_rows=False
    return bicluster_grid



Bicluster_CS1 = bicluster(CS1, 'complete', 'euclidean')



#%% CASE STUDY 2 - HC

#Process Data
import numpy as np
import pandas as pd

#Read the CSV file into a DataFrame

def read_data(filepath):
    """
    Reads a CSV file from the specified filepath, performs data manipulation, and returns a pandas DataFrame.
    Parameters:
        filepath (str): The path to the CSV file to be read.
    Returns:
        dataframe (pandas DataFrame): A DataFrame containing the data from the CSV file after manipulation.
   """
    dataframe = pd.read_csv(filepath, delimiter=",", header=0, index_col=0)
    #dataframe.drop(dataframe.columns[-3:], axis=1, inplace=True)
    return dataframe

filepath = 'C:/Users/inesm/Documents/Tese/Datasets/GSE91383_CS2_filtrado.csv'
case_study2 = pd.read_csv(filepath, header=0, index_col=0)



# Define what is going to be clustered
def what_to_cluster(data_frame, cluster_by):
    
    """
    Define the data frame to use for clustering based on the cluster_by parameter.

    Parameters:
        data_frame: A pandas DataFrame object containing the data to be clustered.
        cluster_by: A string specifying whether to cluster by 'gene' or 'sample'.

    Returns:
        data_cluster: A pandas DataFrame object representing the data frame to be used for clustering.
    """
    if cluster_by == 'gene':
        data_cluster = data_frame
        # If cluster_by is 'gene', the input data_frame is assigned directly to the data_cluster variable.
        
    if cluster_by == 'sample':
        data_cluster = data_frame.transpose(copy=True)
        # If cluster_by is 'sample', the data_frame is transposed (rows become columns) and the resulting transposed data frame is assigned to the data_cluster variable.

    else:
        print ('error = cluster_by must be gene or sample')
        # If cluster_by has any other value, an error message is printed.
    
    return data_cluster


#Hierarchical Clustering
import scipy
import scipy.cluster.hierarchy as sch
import pandas as pd
from scipy.cluster.hierarchy import _LINKAGE_METHODS, dendrogram, linkage, set_link_color_palette
import fastcluster as fc


# Hierarchical Clustering with a desired method and metric
def hierarchical_matrix(data_frame, method, metric):
    """
    Performs hierarchical clustering on a given data frame using the specified parameters.
    
    Args:
        data_frame (pandas.DataFrame): The input data frame containing the data to be clustered.
        method: A string parameter specifying the linkage method to be used for clustering. 
                The method options are 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'
        metric: A string parameter specifying the distance metric to be used for clustering.
                The metric options are 'euclidean', 'cosine', 'correlation'
    Returns:
        The resulting hierarchical clustering linkage matrix.
    """
    
    #preserve_input= false, usa menos memoria
    
    linkage_matrix = fc.linkage(data_frame, method=method, metric=metric, preserve_input='True')
            
    return linkage_matrix

CS2=what_to_cluster(case_study2, 'sample')
HC_CS2= hierarchical_matrix(CS2, 'complete', 'cosine') #linkage_matrix



import biosppy
from biosppy.clustering import _life_time
from scipy.cluster.hierarchy import fcluster

def hierarchical_partition(linkage_matrix, criterion, n):
    """
    Performs hierarchical partitioning on a given linkage matrix.

    Parameters:
        linkage_matrix (ndarray): The linkage matrix obtained from hierarchical clustering.
        criterion (str): The criterion for partitioning. It can be either 'maxcluster' or 'life_time'.
        n (float): Number of clusters to form or life time if 0.

    Returns:
        threshold: Float that corresponds to the threshold to form n number of clusters.
        labels: An array of labels indicating the cluster assignments for each sample.
    """
    if  criterion == 'maxcluster':
        th=((linkage_matrix[-n,2]+linkage_matrix[-(n-1),2])/2)
        labels = fcluster(linkage_matrix, n, criterion='maxclust')
        
    if  criterion == 'life_time':
        N = (len(linkage_matrix)+1)

        if N < 3:
            raise ValueError("The number of objects N must be greater then 2.")

        # compute differences from Z distances
        df = np.diff(linkage_matrix[:, 2])
        # find maximum difference
        idx_max = np.argmax(df)
        mx_dif = df[idx_max]
        # find minimum difference
        mi_dif = np.min(df[np.nonzero(df != 0)])

            
        # find threshold link distance
        th_link = linkage_matrix[idx_max, 2]
        # links above threshold
        idxs = linkage_matrix[np.nonzero(linkage_matrix[:, 2] > th_link)[0], 2]
        #number of links above threshold +1 = number of clusters and singletons
        cont = len(idxs) + 1

        # condition (perceber melhor)
        if mi_dif != mx_dif:
            if mx_dif < 2 * mi_dif:
                cont = 1

        if cont > 1:
            labels = fcluster(linkage_matrix, cont, 'maxclust')
        else:
            labels = np.arange(N, dtype='int')
 
        
        th = ((linkage_matrix[idx_max, 2]+linkage_matrix[idx_max+1,2])/2)
    
    return {'threshold':th, 'samples_labels':labels}

HC_partition_CS2 = hierarchical_partition(HC_CS2, 'life_time',5)


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sb
from seaborn import color_palette


def plot_dendrogram(linkage_matrix, sample_names, cluster_threshold, cmap): # plot the hierarchical clustering
    """
   Plot a Dendrogram
    
    This function generates and displays a dendrogram plot based on the provided linkage matrix,
    which represents the hierarchical clustering of data points. The dendrogram illustrates the
    hierarchical structure of clusters in the data, with vertical lines indicating cluster
    mergers at different levels of similarity.
    
    Parameters:
    - linkage_matrix (array-like): The linkage matrix resulting from hierarchical clustering,
      defining how clusters are merged.
    - labels (list or array-like): Labels or identifiers for the data points being clustered.
    - cluster_threshold (float): A threshold value to color clusters above it differently,
      aiding in the identification of meaningful clusters.
    - cmap (str or colormap, optional): The colormap used for coloring clusters.
    
    Returns:
    - None: The function displays the dendrogram plot but does not return any values.
    
    """
    plt.figure(figsize=(22,8), dpi=300)
    plt.title("Complete Linkage", fontsize=20)
    #plt.xlabel("Samples")
    plt.ylabel("Cosine Distance", fontsize=20)
    sbcmap = sb.color_palette()
    set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in sbcmap])
    dendrogram(Z=linkage_matrix, color_threshold=cluster_threshold, labels=sample_names, 
               above_threshold_color='#b3b3b3', leaf_rotation=70, leaf_font_size=16) #codigo do cinzento
    
    return plt.show()


sbcmap = sb.color_palette("PiYG")
dendogram_CS2 = plot_dendrogram(HC_CS2, CS2.index, HC_partition_CS2['threshold'], sbcmap)



from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

def evaluate_clustering(data_frame, labels): # Evaluate the performed clusters
    """
    Evaluate the quality of hierarchical clustering results using multiple evaluation metrics.

    Parameters:
        data_frame: A pandas DataFrame object containing the data used for clustering.
        labels: The resulting hierarchical clustering linkage matrix.

    Returns:
        A dictionary containing the evaluation scores:
        - 'calinski_harabasz': The Calinski-Harabasz Index.
        - 'davies_bouldin': The Davies-Bouldin Index.
        - 'silhouette': The Silhouette Coefficient.
    """

    # Calculate evaluation scores
    scores = {}

    # Calinski-Harabasz Index
    ch_score = calinski_harabasz_score(data_frame, labels)
    scores['calinski_harabasz'] = ch_score

    # Davies-Bouldin Index
    db_score = davies_bouldin_score(data_frame, labels)
    scores['davies_bouldin'] = db_score

    # Silhouette Coefficient
    silhouette_avg = silhouette_score(data_frame, labels)
    scores['silhouette'] = silhouette_avg

    return scores


scores_CS2 = evaluate_clustering(CS2, HC_partition_CS2['samples_labels'] )

#%% CASE STUDY 2 - Bi-clustering

import seaborn as sb

def bicluster(data, method, metric, figsize=(20, 20), dendrogram_ratio=(0.2, 0.2), cmap='Spectral_r'):
    """
    Performs biclustering on the given data using the specified parameters and visualizes the results.

    Parameters:
        data (pandas.DataFrame or numpy.ndarray): The input data to be biclustered.
        method (str): The linkage method to be used for clustering. Available options are:
                      - 'single', complete','average','weighted','centroid','median','ward'.
        metric (str): The distance metric to be used for clustering. Available options are:
                      - 'euclidean','cosine','correlation'
        figsize (tuple, optional): The figure size for the resulting clustermap. Defaults to (20, 20).
        dendrogram_ratio (tuple, optional): The ratio of the dendrogram sizes. Defaults to (0.2, 0.2).
        cmap (str or colormap, optional): The colormap to be used for the resulting clustermap. Defaults to 'Spectral_r'.

    Returns:
        seaborn.matrix.ClusterGrid: The resulting clustermap object.
    """
    sb.set_theme(color_codes=True)
    bicluster_grid = sb.clustermap(data, method=method, metric=metric, 
                                   figsize=figsize, dendrogram_ratio=dendrogram_ratio, cmap=cmap)
    return bicluster_grid

BC_CS2=bicluster(CS2,'average','correlation')


#%% CASE STUDY 3 - HC

#Process Data
import numpy as np
import pandas as pd

#Read the CSV file into a DataFrame

def read_data(filepath):
    """
    Reads a CSV file from the specified filepath, performs data manipulation, and returns a pandas DataFrame.
    Parameters:
        filepath (str): The path to the CSV file to be read.
    Returns:
        dataframe (pandas DataFrame): A DataFrame containing the data from the CSV file after manipulation.
   """
    dataframe = pd.read_csv(filepath, delimiter=",", header=0, index_col=0)
    #dataframe.drop(dataframe.columns[-3:], axis=1, inplace=True)
    return dataframe

filepath = 'C:/Users/inesm/Documents/Tese/Datasets/GSE162277_CS3_filtrado.csv'
case_study3 = pd.read_csv(filepath, header=0, index_col=0)



# Define what is going to be clustered
def what_to_cluster(data_frame, cluster_by):
    
    """
    Define the data frame to use for clustering based on the cluster_by parameter.

    Parameters:
        data_frame: A pandas DataFrame object containing the data to be clustered.
        cluster_by: A string specifying whether to cluster by 'gene' or 'sample'.

    Returns:
        data_cluster: A pandas DataFrame object representing the data frame to be used for clustering.
    """
    if cluster_by == 'gene':
        data_cluster = data_frame
        # If cluster_by is 'gene', the input data_frame is assigned directly to the data_cluster variable.
        
    if cluster_by == 'sample':
        data_cluster = data_frame.transpose(copy=True)
        # If cluster_by is 'sample', the data_frame is transposed (rows become columns) and the resulting transposed data frame is assigned to the data_cluster variable.

    else:
        print ('error = cluster_by must be gene or sample')
        # If cluster_by has any other value, an error message is printed.
    
    return data_cluster


#Hierarchical Clustering
import scipy
import scipy.cluster.hierarchy as sch
import pandas as pd
from scipy.cluster.hierarchy import _LINKAGE_METHODS, dendrogram, linkage, set_link_color_palette
import fastcluster as fc


# Hierarchical Clustering with a desired method and metric
def hierarchical_matrix(data_frame, method, metric):
    """
    Performs hierarchical clustering on a given data frame using the specified parameters.
    
    Args:
        data_frame (pandas.DataFrame): The input data frame containing the data to be clustered.
        method: A string parameter specifying the linkage method to be used for clustering. 
                The method options are 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'
        metric: A string parameter specifying the distance metric to be used for clustering.
                The metric options are 'euclidean', 'cosine', 'correlation'
    Returns:
        The resulting hierarchical clustering linkage matrix.
    """
    
    #preserve_input= false, usa menos memoria
    
    linkage_matrix = fc.linkage(data_frame, method=method, metric=metric, preserve_input='True')
            
    return linkage_matrix

CS3=what_to_cluster(case_study3, 'sample')
HC_CS3= hierarchical_matrix(CS3, 'average', 'euclidean') #linkage_matrix



import biosppy
from biosppy.clustering import _life_time
from scipy.cluster.hierarchy import fcluster

def hierarchical_partition(linkage_matrix, criterion, n):
    """
    Performs hierarchical partitioning on a given linkage matrix.

    Parameters:
        linkage_matrix (ndarray): The linkage matrix obtained from hierarchical clustering.
        criterion (str): The criterion for partitioning. It can be either 'maxcluster' or 'life_time'.
        n (float): Number of clusters to form or life time if 0.

    Returns:
        threshold: Float that corresponds to the threshold to form n number of clusters.
        labels: An array of labels indicating the cluster assignments for each sample.
    """
    if  criterion == 'maxcluster':
        th=((linkage_matrix[-n,2]+linkage_matrix[-(n-1),2])/2)
        labels = fcluster(linkage_matrix, n, criterion='maxclust')
        
    if  criterion == 'life_time':
        N = (len(linkage_matrix)+1)

        if N < 3:
            raise ValueError("The number of objects N must be greater then 2.")

        # compute differences from Z distances
        df = np.diff(linkage_matrix[:, 2])
        # find maximum difference
        idx_max = np.argmax(df)
        mx_dif = df[idx_max]
        # find minimum difference
        mi_dif = np.min(df[np.nonzero(df != 0)])

            
        # find threshold link distance
        th_link = linkage_matrix[idx_max, 2]
        # links above threshold
        idxs = linkage_matrix[np.nonzero(linkage_matrix[:, 2] > th_link)[0], 2]
        #number of links above threshold +1 = number of clusters and singletons
        cont = len(idxs) + 1

        # condition (perceber melhor)
        if mi_dif != mx_dif:
            if mx_dif < 2 * mi_dif:
                cont = 1

        if cont > 1:
            labels = fcluster(linkage_matrix, cont, 'maxclust')
        else:
            labels = np.arange(N, dtype='int')
 
        
        th = ((linkage_matrix[idx_max, 2]+linkage_matrix[idx_max+1,2])/2)
    
    return {'threshold':th, 'samples_labels':labels}

HC_partition_CS3 = hierarchical_partition(HC_CS3, 'life_time',5)


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sb
from seaborn import color_palette


def plot_dendrogram(linkage_matrix, sample_names, cluster_threshold, cmap): # plot the hierarchical clustering
    """
   Plot a Dendrogram
    
    This function generates and displays a dendrogram plot based on the provided linkage matrix,
    which represents the hierarchical clustering of data points. The dendrogram illustrates the
    hierarchical structure of clusters in the data, with vertical lines indicating cluster
    mergers at different levels of similarity.
    
    Parameters:
    - linkage_matrix (array-like): The linkage matrix resulting from hierarchical clustering,
      defining how clusters are merged.
    - labels (list or array-like): Labels or identifiers for the data points being clustered.
    - cluster_threshold (float): A threshold value to color clusters above it differently,
      aiding in the identification of meaningful clusters.
    - cmap (str or colormap, optional): The colormap used for coloring clusters.
    
    Returns:
    - None: The function displays the dendrogram plot but does not return any values.
    
    """
    plt.figure(figsize=(7,5), dpi=500)
    plt.title("Average Linkage")
    plt.xlabel("Samples")
    plt.ylabel("Euclidean Distance")
    sbcmap = sb.color_palette()
    set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in sbcmap])
    dendrogram(Z=linkage_matrix, color_threshold=cluster_threshold, labels=sample_names, 
               above_threshold_color='#b3b3b3', leaf_rotation=45, leaf_font_size=10) #codigo do cinzento
    
    return plt.show()


sbcmap = sb.color_palette("PiYG")
dendogram_CS3 = plot_dendrogram(HC_CS3, CS3.index, HC_partition_CS3['threshold'], sbcmap)



from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

def evaluate_clustering(data_frame, labels): # Evaluate the performed clusters
    """
    Evaluate the quality of hierarchical clustering results using multiple evaluation metrics.

    Parameters:
        data_frame: A pandas DataFrame object containing the data used for clustering.
        linkage_matrix: The resulting hierarchical clustering linkage matrix.

    Returns:
        A dictionary containing the evaluation scores:
        - 'calinski_harabasz': The Calinski-Harabasz Index.
        - 'davies_bouldin': The Davies-Bouldin Index.
        - 'silhouette': The Silhouette Coefficient.
    """

    # Calculate evaluation scores
    scores = {}

    # Calinski-Harabasz Index
    ch_score = calinski_harabasz_score(data_frame, labels)
    scores['calinski_harabasz'] = ch_score

    # Davies-Bouldin Index
    db_score = davies_bouldin_score(data_frame, labels)
    scores['davies_bouldin'] = db_score

    # Silhouette Coefficient
    silhouette_avg = silhouette_score(data_frame, labels)
    scores['silhouette'] = silhouette_avg

    return scores


scores_CS3 = evaluate_clustering(CS3, HC_partition_CS3['samples_labels'] )

#%% CASE STUDY 3 - Bi-clustering

import seaborn as sb

def bicluster(data, method, metric, figsize=(25, 25), dendrogram_ratio=(0.2, 0.2), cmap='Spectral_r'):
    """
    Performs biclustering on the given data using the specified parameters and visualizes the results.

    Parameters:
        data (pandas.DataFrame or numpy.ndarray): The input data to be biclustered.
        method (str): The linkage method to be used for clustering. Available options are:
                      - 'single', complete','average','weighted','centroid','median','ward'.
        metric (str): The distance metric to be used for clustering. Available options are:
                      - 'euclidean','cosine','correlation'
        figsize (tuple, optional): The figure size for the resulting clustermap. Defaults to (20, 20).
        dendrogram_ratio (tuple, optional): The ratio of the dendrogram sizes. Defaults to (0.2, 0.2).
        cmap (str or colormap, optional): The colormap to be used for the resulting clustermap. Defaults to 'Spectral_r'.

    Returns:
        seaborn.matrix.ClusterGrid: The resulting clustermap object.
    """
    sb.set_theme(color_codes=True)
    bicluster_grid = sb.clustermap(data, method=method, metric=metric, 
                                   figsize=figsize, dendrogram_ratio=dendrogram_ratio, 
                                   cmap=cmap)
    
    
    return bicluster_grid

BC_CS3=bicluster(CS3,'complete','euclidean')


