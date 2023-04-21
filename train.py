import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src import model

def main():
    # pre-processing
    img_path = 'C:\\Users\\Harsh Pal\\Desktop\\Sem_10\\GNR602\\Project\\GNR602-2023-Project\\images\\00128.jpg'
    img = mpimg.imread(img_path)
    pixels = img.reshape((-1,3))

    # fitting
    n_clusters = 2
    agglo = model.AgglomerativeClustering(k=n_clusters, initial_k=25)
    agglo.fit(pixels)

    # prediction
    seg_img = [[agglo.predict_center(list(pixel)) for pixel in row] for row in img]
    seg_img = np.array(seg_img, np.uint8)

    # plotting results
    plt.figure(figsize=(15,15))

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original image')

    plt.subplot(1,2,2)
    plt.imshow(seg_img)
    plt.axis('off')
    plt.title(f'Segmented image with k={n_clusters}')
    plt.show()


if __name__ == "__main__":
    main()
