import numpy as np
import matplotlib.pyplot as plt

# Your original sizes
original_sizes = [(3842, 3842), (3810, 3810), (3780, 3780), (3752, 3752), (3722, 3722), (3692, 3692), (3662, 3662), (3632, 3632), (3602, 3602), (3572, 3572), (3542, 3542), (3512, 3512), (3482, 3482), (3454, 3454), (3422, 3422), (3392, 3392), (3362, 3362), (3332, 3332), (3304, 3304), (3274, 3274), (3244, 3244), (3214, 3214), (3184, 3184), (3154, 3154), (3124, 3124), (3094, 3094), (3064, 3064), (3034, 3034), (3006, 3006), (2974, 2974), (2944, 2944), (2914, 2914), (2884, 2884), (2856, 2856), (2826, 2826), (2796, 2796), (2766, 2766), (2736, 2736), (2706, 2706), (2676, 2676), (2646, 2646), (2616, 2616), (2586, 2586), (2558, 2558), (2526, 2526), (2496, 2496), (2466, 2466), (2436, 2436), (2408, 2408), (2378, 2378), (2348, 2348), (2318, 2318), (2288, 2288), (2258, 2258), (2228, 2228), (2198, 2198), (2168, 2168), (2138, 2138), (2108, 2108), (2080, 2080)]

# Unzip to separate width and height (in this case they are the same)
sizes = np.array(original_sizes)
widths = sizes[:,0]

# Define moving average function
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Apply moving average
window_size = 5  # Set the size of the moving window
smooth_widths = moving_average(widths, window_size)

# Plot old and new sizes
plt.plot(widths, label='Original')
plt.plot(smooth_widths, label='Smoothed')
plt.legend()
plt.show()
