import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_bbox(binary_mask, bbox):
    y1 = bbox[0]
    x1 = bbox[1]
    y2 = bbox[2]
    x2 = bbox[3]

    # Create a figure and axis
    fig, ax = plt.subplots()
    # Display the binary mask
    ax.imshow(binary_mask, cmap='binary', interpolation='none')
    # Create a rectangle patch using the bounding box coordinates
    rect = patches.Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1 + 1, y2 - y1 + 1,
                            linewidth=2, edgecolor='r', facecolor='none')

    # Add the rectangle patch to the plot
    ax.add_patch(rect)
    # Set axis limits and show the plot
    ax.set_xlim(0, binary_mask.shape[1])
    ax.set_ylim(0, binary_mask.shape[0])
    plt.show()


