# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:27:31 2023

@author: inesmpsoares
"""

#%% PROCESS DATA
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
    dataframe.drop(dataframe.columns[-3:], axis=1, inplace=True)
    return dataframe


def build_data(dataframe, delimiter="_", name_nomenclature=["protocol", "day", "replication"]):
    """
    Builds a new data frame by reorganizing the input dataframe based on specified parameters.

    Parameters:
        dataframe (pandas.DataFrame): The input data frame to be reorganized.
        delimiter (str, optional): The delimiter used in the column names to split and extract additional information.
                                   Defaults to "_".
        name_nomenclature (list, optional): A list of strings specifying the names to assign to the split columns.
                                            The order of names should match the order of splits obtained from the column names.
                                            Defaults to ["protocol", "day", "replication"].

    Returns:
        pandas.DataFrame: The reorganized data frame with columns 'expression', 'gene', and split columns from the name nomenclature.
    """

    new_df = pd.DataFrame()
    new_df['expression'] = np.hstack(dataframe.values)
    new_df['gene'] = np.repeat(dataframe.index, dataframe.shape[1])
    new_df['name'] = np.tile(dataframe.columns, dataframe.shape[0])
    new_df[name_nomenclature] = new_df.name.str.split(delimiter, expand=True)
    new_df.drop("name", axis=1, inplace=True)
    
    return new_df


def perform_custom_groupby(dataframe, groupby_columns):
    """
    Performs a groupby operation on a DataFrame based on the specified columns.

    Parameters:
        dataframe (pandas.DataFrame): The DataFrame to be grouped.
        groupby_columns (list): A list of column names to group the DataFrame by.

    Returns:
        pandas.DataFrame: The resulting grouped DataFrame.
    """
    grouped_df = dataframe.groupby(by=groupby_columns).mean().reset_index()
    return grouped_df



def return_to_data(data_frame):
    """
    Performs grouping, mean calculation, pivoting, and column renaming on the given data frame, to return the data frame into the original structure.

    Parameters:
        data_frame: The input data frame containing the groupby data frame with the gene expression data.

    Returns:
        A processed data frame with grouped, averaged, pivoted, and renamed columns.
    """

    returned_df = data_frame.pivot(index="gene", columns=[col for col in data_frame.columns if col not in ["gene", "expression"]], values="expression")
    returned_df.columns = returned_df.columns.map("_".join)
    
    return returned_df


# %% HIERARCHICAL CLUSTERING
import scipy
import scipy.cluster.hierarchy as sch
import pandas as pd
from scipy.cluster.hierarchy import _LINKAGE_METHODS, dendrogram, linkage, set_link_color_palette
import fastcluster as fc

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


# Hierarchical Clustering of all methods and metrics

def hierarchical_matrix_all(data_frame, methods, metrics):
    """
    Performs hierarchical clustering on a given data frame using the specified parameters.
    
    Parameters:
        data_frame (pandas.DataFrame): The input data frame containing the data to be clustered.
        methods (list): A list of string parameters specifying the linkage methods to be used for clustering.
                        Available method options are 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'.
        metrics (list): A list of string parameters specifying the distance metrics to be used for clustering.
                        Available metric options are 'euclidean', 'cosine', 'correlation'.
                        
    Returns:
        A dictionary containing the resulting hierarchical clustering linkage matrices, with keys representing
        the combination of method and metric used.
    """
    linkage_matrices = {}
    
    for method in methods:
        for metric in metrics:
            if (method == 'centroid' or method == 'median' or method == 'ward') and metric != 'euclidean':
                # Skip combination if method is 'centroid', 'median', or 'ward' but metric is not 'euclidean'
                continue
            linkage_matrix = fc.linkage(data_frame, method=method, metric=metric, preserve_input='True')
            key = f"{method}_{metric}"
            linkage_matrices[key] = linkage_matrix
    
    return linkage_matrices


# Hierarchical Partiton of the clusters

import biosppy
from biosppy.clustering import _life_time
from scipy.cluster.hierarchy import fcluster
#The 't' parameter represents the distance threshold that determines the maximum dissimilarity allowed for 
#two observations to be assigned to the same cluster. A smaller 't' value will result in more clusters and
#potentially finer-grained clustering, while a larger 't' value will lead to fewer clusters and potentially 
#coarser-grained clustering. You can experiment with different threshold values to find the clustering granularity
#that best suits your data and problem domain. Adjusting the 't' parameter allows you to control the trade-off 
#between having more detailed clusters versus larger, more generalized clusters.

def hierarchical_partition_labels(linkage_matrix, criterion, n):
    """
    Performs hierarchical partitioning on a given linkage matrix.

    Parameters:
        linkage_matrix (ndarray): The linkage matrix obtained from hierarchical clustering.
        criterion (str): The criterion for partitioning. It can be either 'maxcluster' or 'life_time'.
        n (float): The threshold parameter for partitioning. Only required for the 'maxcluster' criterion.

    Returns:
        labels: An array of labels indicating the cluster assignments for each sample.
    """
    
    if  criterion == 'maxcluster':
        labels = fcluster(linkage_matrix, n, criterion='maxclust')
        
    if  criterion == 'life_time':
        N = (len(linkage_matrix)+1)
        labels = _life_time(linkage_matrix, N)
    
    return labels


def hierarchical_partition_threshold(linkage_matrix, n):
    """
    Performs hierarchical partitioning on a given linkage matrix.

    Parameters:
        linkage_matrix (ndarray): The linkage matrix obtained from hierarchical clustering.
        n (float): Number of clusters to form or life time if 0.

    Returns:
        threshold: Float that corresponds to the threshold to form n number of clusters.
    """
    
    if  n == 0:
        df = np.diff(linkage_matrix[:, 2])
        # find maximum difference
        idx_max = np.argmax(df)
        # find threshold
        th = ((linkage_matrix[idx_max, 2]+linkage_matrix[idx_max+1,2])/2)
        
    else  :
        th=((linkage_matrix[-n,2]+linkage_matrix[-(n-1),2])/2)
    
    return th



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


#%% CLUSTERS EVALUATION

# Evaluate the performed clusters

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

def evaluate_clustering(data_frame, labels):
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


#%% PLOT HIERARCHICAL CLUSTERING

# plot the hierarchical clustering
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_dendrogram(linkage_matrix, labels, cluster_threshold, cmap):
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
    plt.title("Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    sbcmap = sb.color_palette(cmap, labels)
    set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in sbcmap])
    dendrogram(Z=linkage_matrix, color_threshold=cluster_threshold, labels=labels, above_threshold_color='#b3b3b3')
    
    return plt.show()



#%% BI-CLUSTERIRNG

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


