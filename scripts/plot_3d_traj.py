import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

def read_lines_as_strings(file_path):
    """
    Reads the specified text file line by line and returns a list of lines as strings.
    """
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Strip trailing newline and surrounding whitespace
                lines.append(line.strip())
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except IOError:
        print(f"Error: Unable to read file '{file_path}'.")
    
    return lines

# First trajectory file (comment out if not needed)
# data_lines = read_lines_as_strings('/home/samqiao/ASRL/vtr3_testing/result/radar/jan29_parking_t6/traj/teach_traj_estimated.txt')

# Second trajectory file (this overwrites data_lines if both are run)
def plot_3d_traj(file_path):
    data_lines = read_lines_as_strings(file_path)

    print(f"Read {len(data_lines)} lines from the file.")
    # Extract X, Y, Z coordinates
    x, y, z = [], [], []
    for line in data_lines:
        parts = line.split(',')
        # print(parts)
        # parts[1], parts[2], parts[3] should correspond to X, Y, Z
        x.append(float(parts[1]))
        y.append(float(parts[2]))
        z.append(float(parts[3]))

    # Calculate total path length
    path_length = 0.0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        dz = z[i] - z[i - 1]
        path_length += math.sqrt(dx*dx + dy*dy + dz*dz)

    print(f"Total path length: {path_length:.3f} units")

    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with color representing time progression
    scatter = ax.scatter(x, y, z, c=np.arange(len(x)), cmap='viridis', s=10, label='Trajectory Points')
    ax.plot(x, y, z, color='gray', alpha=0.4, linewidth=0.8, label='Path')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()

    # Add a color bar to show the progression over time
    plt.colorbar(scatter, label='Time Step', pad=0.1)

    plt.title('3D Trajectory Visualization')
    plt.show()
