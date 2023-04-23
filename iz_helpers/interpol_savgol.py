import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Define the values for width, mask_width, and num_interpol_frames
width = 512
mask_width = round(width*0.25)
num_interpol_frames = 30

# Create a list to store the values of interpol_width for each frame
interpol_width_values = []

# Calculate interpol_width for each frame and add it to the list
for j in range(num_interpol_frames):
    interpol_width = round((1 - (1 - 2 * mask_width / width)**(1 - (j + 1) / num_interpol_frames)) * width / 2)
    interpol_width_values.append(interpol_width)

# Apply a SAVGOL filter to the interpol_width values
window_size = 7  # adjust the window size as needed
interpol_width_smoothed = savgol_filter(interpol_width_values, window_size, 2)

inter_scalar = np.round(interpol_width_smoothed).astype(int) # Round the output to the nearest integer


# Plot the original and smoothed values of interpol_width on a graph
plt.plot(range(1, num_interpol_frames + 1), interpol_width_values, label='Original')
plt.plot(range(1, num_interpol_frames + 1), interpol_width_smoothed, label='Smoothed')
plt.plot(range(1, num_interpol_frames + 1), inter_scalar, label='SmoothedInt')
plt.xlabel('Frame Number')
plt.ylabel('Interpolation Width')
plt.title('Dolly-Out Animation Graph')
plt.legend()
plt.show()
