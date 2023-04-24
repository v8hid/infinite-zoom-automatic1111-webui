import matplotlib.pyplot as plt
import math

# Define the values for width, mask_width, and num_interpol_frames
width = 512
mask_width = round(width * 0.25)
num_interpol_frames = 30


def exponential_interpolation(b, t):
    exponent_coeff = 1  # between 0 to 1
    t = t / b
    y = math.pow(b, math.pow(t, exponent_coeff))
    return y


# Calculate the logarithmic scaling factor
log_scale_factor = math.log(width / mask_width) / (num_interpol_frames - 1)

# Create a list to store the values of interpol_width for each frame
interpol_width_values = []

# Calculate interpol_width for each frame and add it to the list
for j in range(num_interpol_frames):
    # scale_factor = math.exp(-j*log_scale_factor)
    # interpol_width = round(width*scale_factor/2)
    interpol_width = exponential_interpolation(
        mask_width, (num_interpol_frames - j) * mask_width / num_interpol_frames
    )
    interpol_width_values.append(interpol_width)
# Plot the values of interpol_width on a graph
plt.plot(range(1, num_interpol_frames + 1), interpol_width_values)
plt.xlabel("Frame Number")
plt.ylabel("Interpolation Width")
plt.title("Dolly-Out Animation Graph")
plt.show()
