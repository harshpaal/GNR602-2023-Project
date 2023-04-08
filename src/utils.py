import numpy as np
import matplotlib.pyplot as plt

def cluster_and_plot(n_clusters, model, pixels, img):
    agglo = model(k=n_clusters, initial_k=25)
    agglo.fit(pixels)

    new_img = [[agglo.predict_center(pixel) for pixel in row] for row in img]
    new_img = np.array(new_img, np.uint8)

    plt.figure(figsize=(15,15))

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original image')

    plt.subplot(1,2,2)
    plt.imshow(new_img)
    plt.axis('off')
    plt.title(f'Segmented image with k={n_clusters}')

    plt.show()

def euclidean_distance(point1, point2):
    """
    Computes euclidean distance of point1 and point2.
    
    point1 and point2 are lists.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def clusters_distance(cluster1, cluster2):
    """
    Computes distance between two clusters.
    
    cluster1 and cluster2 are lists of lists of points
    """
    return max([euclidean_distance(point1, point2) for point1 in cluster1 for point2 in cluster2])
  
def clusters_distance_2(cluster1, cluster2):
    """
    Computes distance between two centroids of the two clusters
    
    cluster1 and cluster2 are lists of lists of points
    """
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)