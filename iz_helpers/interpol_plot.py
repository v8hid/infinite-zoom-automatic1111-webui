import matplotlib.pyplot as plt

# Define the values for width, mask_width, and num_interpol_frames
width = 512
mask_width = round(512*0.25)
num_interpol_frames = 30

# Create a list to store the values of interpol_width for each frame
interpol_width_values = []

# Calculate interpol_width for each frame and add it to the list
for j in range(num_interpol_frames):
    interpol_width = int((1 - (1 - 2 * mask_width / width)**(1 - (j + 1) / num_interpol_frames)) * width / 2)
    interpol_width_values.append(interpol_width)

# Plot the values of interpol_width on a graph
plt.plot(range(1, num_interpol_frames + 1), interpol_width_values)
plt.xlabel('Frame Number')
plt.ylabel('Interpolation Width')
plt.title('Dolly-Out Animation Graph')
plt.show()




