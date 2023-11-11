# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 10:36:56 2023

@author: inesm
"""


#MiniSom

#imports

from minisom import MiniSom
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, axis, show, pcolor, colorbar, bone

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

#Data transformation

filepath = 'Data_directory.csv'

read_data = pd.read_csv(filepath, delimiter=",", header=0, index_col=0)
d = what_to_cluster(read_data, 'gene')

# Convert the DataFrame to a NumPy array to be used in SOM algorithm
data = d.to_numpy()




#SOM initialization
def init_som(data, grid_rows, grid_columns, sigma, learning_rate):
    """
    Initialize a Self-Organizing Map (SOM) with the specified hyperparameters.

    Parameters:
    - data: The input data for initialization.
    - grid_rows: Number of rows in the SOM grid.
    - grid_columns: Number of columns in the SOM grid.
    - sigma: The sigma parameter for SOM initialization.
    - learning_rate: The learning rate for SOM initialization.

    Returns:
    - som: The initialized SOM instance.
    """
    som = MiniSom(x=40, y=40, input_len=data.shape[1], sigma=sigma, 
                  learning_rate=learning_rate) #guassian function as default for neighborhood function
    som.pca_weights_init(data)
    return som




#SOM training
def train_som(som, data, iterations):
    """
    Train a Self-Organizing Map (SOM) with the specified number of iterations, 
    print the training time, and return the matrix of weights.

    Parameters:
    - som: The initialized SOM instance.
    - data: The input data for training.
    - iterations: Number of training iterations.

    Returns:
    - som_matrix: The matrix of weights of the trained SOM.
    """
    start_time = time.time()
    som.train(data, iterations, use_epochs='True',random_order='False')
    elapsed_time = time.time() - start_time
    som_matrix = som.get_weights()  #Obtain the matrix of the weights of the neural network
    print("Training time:", elapsed_time, "seconds")
    
    return som_matrix


#re-shape SOM matrix

def reshape_som_matrix(som_matrix):
    """
    Reshape a 3D self-organizing map (SOM) matrix into a 2D matrix by merging the first two dimensions into one.

    Args:
        som_matrix (numpy.ndarray): The 3D SOM matrix to reshape.

    Returns:
        numpy.ndarray: The reshaped 2D matrix to perform k-means
    """
    reshaped_kmeans_matrix = som_matrix.reshape(-1, som_matrix.shape[-1])
    return reshaped_kmeans_matrix


#Save SOM train
import pickle

def save_training(matrix, output_file):
    """
    Save a som training matrix to a CSV or pickle file based on its dimensionality.

    Parameters:
    - matrix: The matrix to be saved.
    - output_file: The name of the output file.
    """
    if len(matrix.shape) == 2:
        # If the matrix is 2D, save it to a CSV file
        np.savetxt(output_file, matrix, delimiter=',')
        print(f"Saved 3D som matrix to CSV: {output_file}")
    elif len(matrix.shape) == 3:
        # If the matrix is 3D, save it to a pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(matrix, f)
        print(f"Saved 2D som matrix to pickle: {output_file}")
    else:
        print("Unsupported matrix dimensionality. Please provide a 2D or 3D matrix.")


#SOM Errors 

def som_errors(som_trained, data):
    """
    Calculate quantization error and topographic error for SOM.

    Parameters:
    - som: The SOM instance.
    - data: The data set used to calculate errors.

    Prints:
    - quantization_error: The quantization error.
    - topographic_error: The topographic error.
    """
    quantization_error = som.quantization_error(data)
    topographic_error = som.topographic_error(data)
    
    print("Quantization Error:", quantization_error)
    print("Topographic Error:", topographic_error)



#SOM Portraits
import numpy as np
import matplotlib.pyplot as plt

def portrait_sample(som_trained, sample_index):
    """
    Plot a specific sample from the SOM matrix using matshow.

    Parameters:
    - som_matrix: The matrix of weights of the trained SOM.
    - sample_index: Index of the sample to be plotted.
    """
    sample = som_matrix[:, :, sample_index]
    plt.matshow(sample, cmap='jet')  # Adjust the colormap as needed
    plt.title(f'Sample {sample_index}')
    plt.colorbar()
    plt.show()



def portrait_sample_mean(som_matrix, sample_indices):
    """
    Plot the mean portrait of specific samples from the SOM matrix using matshow.

    Parameters:
    - som_matrix: The matrix of weights of the trained SOM.
    - sample_indices: List of indices of the samples to be plotted.
    """
    # Initialize an empty array to store the combined samples
    combined_sample = np.zeros_like(som_matrix[:, :, 0])

    # Calculate the mean of the selected samples
    for index in sample_indices:
        combined_sample += som_matrix[:, :, index]
    combined_sample /= len(sample_indices)

    plt.matshow(combined_sample, cmap='jet')  # Adjust the colormap as needed
    #plt.title(f'Mean of Samples {sample_indices}')
    plt.title(f'Mean of hiPSC Samples')
    plt.colorbar()
    plt.show()


def portrait_all(som_trained, sample_names):
    """    Plot all samples in the SOM matrix in a grid of subplots with sample names as titles,
    and include a colorbar for each subplot using matshow.

    Parameters:
    - som_matrix: The matrix of weights of the trained SOM.
    - sample_names: A list of sample names corresponding to each sample.
    """
    num_samples = som_matrix.shape[2]
    num_rows = int(np.sqrt(num_samples))
    num_cols = int(np.ceil(num_samples / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10,10))
    fig.subplots_adjust(hspace=0.5)

    for i in range(num_samples):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        sample = som_matrix[:, :, i]
        im = ax.matshow(sample, cmap='jet', vmin =-1.78, vmax=1.21)  # Adjust the colormap as needed
        ax.set_title(sample_names[i])  # Set the title to the corresponding sample name
        ax.set_xticks([])  # Remove x-axis labels
        ax.set_yticks([]) # Remove y-axis labels
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, ticks=[-1.7,1.2])  # Add a colorbar
        
    # Remove empty subplots if the number of samples is not a perfect square
    for i in range(num_samples, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])
    
    plt.show()


#Distance Map

def plot_distance_map(som_trained):
    """
    Plot the distance map of a Self-Organizing Map (SOM).

    Parameters:
    - som: The trained SOM instance.

    Returns:
    - None
    """
    bone()
    plt.pcolor(som.distance_map().T, cmap='jet')  # Distance map as background
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()

distance_map= plot_distance_map(som_matrix)


#%%

#Gene Search

def get_metagenes_indices(data_array):
    """
    Get metagenes indices using th som.win_map function.

    Parameters:
    data_array: Input data for metagenes computation.

    Returns:
    dict: A dictionary with metagenes as keys and gene positions in the CSV file as values.
    """
    metagenes_indices = som.win_map(data_array, return_indices=True)
    return metagenes_indices


def metagenes_with_geneIDs(metagenes_indices, csv_file):
    """
    Update metagenes indices with gene IDs.

    Parameters:
    metagenes_indices (dict): A dictionary with metagenes as keys and gene positions as values.
    csv_file (DataFrame): DataFrame containing gene data from the CSV file.

    Returns:
    dict: A dictionary with metagenes as keys and gene IDs as values.
    """
    updated_metagenes_indices = {}

    for metagene, positions in metagenes_indices.items():
        gene_ids = [csv_file.iloc[position - 1].name if 1 <= position <= len(csv_file) else "Unknown" for position in positions]
        updated_metagenes_indices[metagene] = gene_ids

    return updated_metagenes_indices



#Save metagenes_IDs

import csv

# Specify the file path
file_path = 'metagenes_IDs.csv'

with open(file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write the header row
    csvwriter.writerow(['Metagene', 'Gene IDs'])
    
    # Write the data
    for metagene, gene_ids in metagenes_IDs.items():
        csvwriter.writerow([metagene, ', '.join(gene_ids)])

import pickle
metagenes_IDs_pickle = open('metagenes_IDs.pkl', 'wb')

pickle.dump(metagenes_IDs, metagenes_IDs_pickle)

metagenes_IDs_pickle.close()


def gene_search_SOM(metagenes_IDs, gene_ids_to_find):
    """
    Find metagenes containing specific a gene ID.

    Parameters:
    metagenes_IDs (dict): A dictionary with metagenes as keys and lists of gene IDs as values.
    gene_ids_to_find (list): A list of gene IDs to search for.

    Returns:
    list: A list of metagenes that contain the specified gene IDs.
    """
    gene_location = []

    for metagene, gene_ids in metagenes_IDs.items():
        if any(gene_id in gene_ids_to_find for gene_id in gene_ids):
            gene_location.append(metagene)

    return gene_location



def portrait_gene_search(som_trained, sample_indices):
    """
    Visualize the gene position of metagene on the SOM grid, for a specific sample.

    Parameters:
    som_trained (numpy.ndarray): A trained Self-Organizing Map.
    sample_index (int): The index of the sample to be visualized.

    Parameters:
    som_matrix (numpy.ndarray): The SOM matrix.
    sample_index (int): The index of the sample to be visualized.

    Returns:
    None
    """
    metagene_keys = gene_location_SOM
    key_y, key_x = zip(*metagene_keys)
    
    combined_sample = np.zeros_like(som_matrix[:, :, 0])

    # Calculate the mean of the selected samples
    for index in sample_indices:
        combined_sample += som_matrix[:, :, index]
    combined_sample /= len(sample_indices)
    
    plt.matshow(combined_sample, cmap='jet',interpolation='none')
    plt.colorbar()
    #plt.title(f'Sample {sample_index}')
    plt.title('Lapatinib_4')
    plt.scatter(key_x, key_y, s=15, c='#000000', marker='o')
    plt.tick_params(right = False , labelbottom = False, bottom = False)
    plt.show()


#%%

#K-means

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Find the optimal K for kmeans clustering using elbow criterion
def find_optimal_k(data_kmeans, max_k=10):
    """
    Find the optimal number of clusters (k) using the Elbow Method.

    Parameters:
      data_kmeans (pd.DataFrame): The data for KMeans clustering.
      max_k (int): The maximum number of clusters to consider (default is 10).

   Returns:
      optimal_k (int): The optimal number of clusters as determined by the Elbow Method, and plot the optimal k.
  """
    sse = {}
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init= 'k-means++',max_iter=1000, n_init=10).fit(data_kmeans)
        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center

    # Plot the SSE values for different values of k
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.title("Elbow Method for Optimal k - Case Study 2")

    # Determine the optimal k by looking for the "elbow" point in the graph
    optimal_k = None
    prev_sse = None
    for k, current_sse in sse.items():
        if prev_sse is not None and prev_sse - current_sse < 0.1 * prev_sse:
            optimal_k = k - 1 
            break
        prev_sse = current_sse

    # Add a circle at the elbow point
    plt.scatter(optimal_k, sse[optimal_k], c='red', marker='o', s=100, label=f'Optimal k ({optimal_k})')

    plt.legend()
    plt.show()
    
    print("Optimal number of clusters (k):", optimal_k)
    return optimal_k


# Perfom k-means clustering
from scipy.cluster.vq import kmeans2

def kmeans_clustering(data, k):
    """
    Perform K-means clustering on a 2D data matrix.

    Parameters:
    data (numpy.ndarray): A 2D array containing the data points for clustering.
    k (int): The number of clusters to form.

    Returns:
    numpy.ndarray: An array of cluster centroids.
    numpy.ndarray: An array of cluster labels for each data point.
    """
    
    # Perform K-means clustering
    centroids, labels = kmeans2(data, k=k, minit='++')
    
    return centroids, labels

#Calculate silhouette score
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(som_matrix_2d, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

 
#Calculate inertia
inertia = np.sum((som_matrix_2d - centroids[cluster_labels]) ** 2)
print(f"Inertia (Within-Cluster Sum of Squares): {inertia}")

#Save cluster labels

import csv 

# Specify the file path
file_path = 'cluster_labels.csv'

# Open the CSV file for writing
with open(file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write the cluster labels to the CSV file
    csvwriter.writerow(["Index", "Cluster Label"])
    for i, label in enumerate(cluster_labels):
        csvwriter.writerow([i, label])

import pickle
cluster_labels_pickle = open('cluster_labels.pkl', 'wb')

pickle.dump(cluster_labels, cluster_labels_pickle)

cluster_labels_pickle.close()
  



def plot_kmeans_clusters(kmeans_labels, k):
    """
    Plot the K-Means clustering results as a matrix using matshow.

    Args:
        kmeans_labels (numpy.ndarray): The cluster labels from K-Means clustering.
        k (int): The number of clusters (k value) used in K-Means clustering.
    """
    sqrt_shape = int(np.sqrt(kmeans_labels.shape[0]))  # Calculate the square root of the number of labels

    labels_2d = kmeans_labels.reshape(sqrt_shape, sqrt_shape)
    labels_2d = labels_2d.transpose()

    # Create a matshow plot of the clustered data
    plt.matshow(labels_2d, cmap='jet')
    plt.title(f'K-Means Clustering with k={k}')

    plt.show()




import numpy as np
import matplotlib.pyplot as plt

def plot_kmeans_clusters_label(kmeans_labels, k, cluster_centers=None):
    """
    Plot the K-Means clustering results as a matrix using matshow.

    Args:
        kmeans_labels (numpy.ndarray): The cluster labels from K-Means clustering.
        k (int): The number of clusters (k value) used in K-Means clustering.
        cluster_centers (numpy.ndarray): (Optional) The cluster centers obtained from K-Means clustering.
    """
    sqrt_shape = int(np.sqrt(kmeans_labels.shape[0]))  # Calculate the square root of the number of labels

    labels_2d = kmeans_labels.reshape(sqrt_shape, sqrt_shape)
    labels_2d = labels_2d.transpose()

    # Create a matshow plot of the clustered data
    plt.matshow(labels_2d, cmap='jet')
    plt.title(f'K-Means Clustering with k={k}')
    
    if cluster_centers is not None:
        for i in range(k):
            y, x = np.where(labels_2d == i)
            plt.text(x.mean(), y.mean(), f'{i}', fontsize=8, color='black')
    
    plt.show()
