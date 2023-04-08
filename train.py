import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src import model, utils

def main():
    img_path = 'C:\\Users\\Harsh Pal\\Desktop\\Sem_10\\GNR602\\Project\\GNR602-2023-Project\\images\\samp.jpg'
    img = mpimg.imread(img_path)
    pixels = img.reshape((-1,3))

    n_clusters = 2
    agglo = model.AgglomerativeClustering(k=n_clusters, initial_k=25)
    agglo.fit(pixels)


if __name__ == "__main__":
    main()
