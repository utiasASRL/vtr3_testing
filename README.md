# VT&R3 Testing for Direct Methods
### The Posegraph Tools
Read from the graph folder to load pose graphs including transformations and data associated with all the vertices.

The basic idea is to replicate the pose graph data structure used in [VT&R](https:/github.com/utiasASRL/vtr3).

Note that the current implementation does not support creating or writing to pose graphs.
This library is useful to load point clouds into PyTorch or open3d for machine learning or visualization offline. 

This library depends on ROS2 for reading the bag files used in VT&R. It is recomended that you run this repository inside of a [Docker container built for VT&R](https://github.com/utiasASRL/vtr3/wiki/Installation).

The samples contain tools for plotting paths. 

There is an optional open3d dependency for visualization. 

To use vtr3_pose_graph you can install with 
```bash
pip3 install -e .
```

To run the samples, after installing
```bash
source $VTRSRC/main/install/setup.bash #For ROS2 and VTR message types
python3 samples/plot_repeat_path.py -g /path/to/graph       
```

## Steps to run the Direct Toy problem 
The main script is located at: scripts/direct/direct_toy.py
But before you can run this, you need to process the posegraph data first whose path is specified in the direct_config.yaml

Note that: You need to source $VTRSRC/main/install/setup.bash before running process_posegraph.py
Also you need to install some dependencies: CV_Bridge; pylgmath and such

### Step 1
Once specify the correct posegraph path
Run 
```bash
source $VTRSRC/main/install/setup.bash
python3 process_posegraph.py
```

This will generate a folder called grassy_t2_r3 with the following structure
```bash
----grassy_t2_r3
    ----direct (stores the result)
    ----teach (stores the extracted teach data)
        ----teach.npz
    ----repeat (stores the extracted repeat data)
        ----repeat.npz
```

### Step 2
You can run 
```bash 
python3 direct_toy.py 
```
directly and it will loop through all the data and generate the results in the direct folder
Essentially, it is storing all the estimated Teach and Repeat Edge into direct/result.npz

### Step 3
You can run the plotter.py to plot the localization and path-tracking error 

## Direct_toy.py is the main file with all the algorithm