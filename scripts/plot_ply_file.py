import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

# Load the PLY file
ply_path = "your_file.ply"  # Replace with your .ply path
pcd = o3d.io.read_point_cloud(ply_path)

# Get the XYZ coordinates as a NumPy array
points = np.asarray(pcd.points)

# Choose a 2D projection — here, XY plane (you can switch to YZ or XZ if needed)
x = points[:, 0]
y = points[:, 1]

# Plot using matplotlib
plt.figure(figsize=(8, 8))
plt.scatter(x, y, s=0.5, color='black')  # You can adjust size and color
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("2D Projection of Point Cloud (XY Plane)")
plt.axis('equal')  # Ensure equal aspect ratio
plt.grid(True)
plt.show()
