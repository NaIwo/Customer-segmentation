import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def silhouette_plot(X, n_clusters, cluster_labels, silhouette_scores, silhouette_samples):
    margin = 10
    plt.figure(figsize = (10, 5))
    ax = plt.axes()
    ax.set_xlim([-0.1, 1.0])
    ax.set_ylim([0, len(X) + (n_clusters + 1) * margin])
    down = margin
    for i in range(n_clusters):
        values = silhouette_samples[cluster_labels == i]
        sorted_values = np.sort(values)
        
        size = np.shape(sorted_values)[0]
        up = down + size
        
        ax.fill_betweenx(np.arange(down, up), 0, sorted_values)
        ax.text(-0.05, down + 0.5 * size, str(i))
        
        down = up + margin
    
    plt.axvline(silhouette_scores, color = 'red', linestyle = ':')
    
    plt.title('silhouette plot with {} clusters'.format(n_clusters))
    
    ax.annotate('Average silhouette score {0:4.2f}'.format(silhouette_scores), xy=(silhouette_scores + .009, (len(X) + (n_clusters + 1) * margin) / 2), 
            xytext=(silhouette_scores + 0.2, (len(X) + (n_clusters + 1) * margin) / 1.5),
            arrowprops=dict(facecolor='black'))
    ax.set_xlabel('Silhouette score')
    ax.set_ylabel('Labels')
    

        
    plt.show()

def draw2D(dataset, labels, loc_str = 'upper right', palette = 'deep'): 
    warnings.simplefilter("ignore")
    
    plt.figure(figsize = (5,5))
    ax = sns.scatterplot(dataset[:, 0], dataset[:, 1], palette = palette, hue = labels, c = labels)
    plt.title("2D decomposition")
    plt.xlabel("Z1")
    plt.ylabel("Z2")  
    plt.legend(loc = loc_str)

def draw3D(dataset, labels): 
    warnings.simplefilter("ignore")
    
    fig = plt.figure(figsize = (12,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], s=60, cmap="Set2_r", c = labels)
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes", loc = 2)
    ax.add_artist(legend1)

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_zlabel("z3")

    ax.set_title("3D decomposition")