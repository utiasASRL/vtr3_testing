# VT&R3 Testing for Dense Radar Resgitration Methods
## Steps to run the Direct Toy problem 
Please clone all the submodules
```bash
git submodule update --init --recursive
```

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

You will also need to generate the ppk gps npz a lot easier. 

They are stored as (n,4) in the order of time, x, y, z where z is a zero array
You need to have the ppk folder path specified correctly and ppk folder should be structured as,

```bash
    ----localization_data
        ----ppk
            ----grassy_t2
                ----grassy_t2.txt
                ----grassy_t2_ros.txt
                ----grassy_t2_gps_fix.csv
                ----grassy2_0_BESTPOS.ASCII
```

Then
```bash
python3 process_ppk_ros.py
```

This will generate a folder called grassy_t2_r3 with the following structure
```bash
----grassy_t2_r3
    ----direct (stores the result)
    ----teach (stores the extracted teach data)
        ----teach.npz
        ----teach_ppk.npz
    ----repeat (stores the extracted repeat data)
        ----repeat.npz
        ----repeat_ppk.npz
```

### Step 2
You can run 
```bash 
python3 direct_toy.py 
```
directly and it will loop through all the data and generate the results in the direct folder
Essentially, it is storing all the estimated Teach and Repeat Edge into direct/result.npz

### Step 3
You can run the 
```bash 
python3 plotter.py 
```
to plot the localization and path-tracking error 

## Direct_toy.py is the main file with all the algorithm



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