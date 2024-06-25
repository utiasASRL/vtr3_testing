# Rebuild vtr_testing_radar
export ASRLROOT=~/ASRL
export ROOTDIR=${ASRLROOT}/vtr3_testing
export VTRROOT=~/ASRL/vtr3         # (INTERNAL default) root directory
export VTRSRC=${VTRROOT}/src       # source code (this repo)
export VTRDATA=${ROOTDIR}/data     # datasets
export VTRTEMP=${VTRROOT}/temp     # default output directory
export VTRMODELS=${VTRROOT}/models # .pt models for TorchScript
export WARTHOG=${VTRROOT}/warthog  # warthog source (this repo)

echo $VTRDATA
 
source ~/.bashrc
source /opt/ros/humble/setup.bash
source ${VTRSRC}/main/install/setup.bash # source the vtr3 environment


cd ${ASRLROOT}/vtr3_testing/test
 # go to where vtr_testing_radar is located
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source ${ASRLROOT}/vtr3_testing/test/install/setup.bash # source the vtr_testing_radar environment

export ODO_INPUT=boreas-2020-11-26-13-58
export LOC_INPUT=boreas-2021-02-02-14-07

# Get arguments
MODE=localization         # [odometry, localization]
SENSOR=radar       # [radar, lidar, radar_lidar]
# ODO_INPUT=$3    # Boreas sequence
# LOC_INPUT=$4    # Boreas sequence, not used if mode=odometry

# Set results subfolder, VTRRESULT is set in setup_container.sh
export VTRRRESULT=${ROOTDIR}/bash_scripts/${SENSOR}
mkdir -p ${VTRRRESULT}

# Load in param file based on sensor
PARAM_FILE=${VTRSRC}/config/navtech_warthog_default.yaml

# Call corresponding script from vtr_testing_radar
# run odometry first
# Log
echo "Running odometry on sequence ${ODO_INPUT}, storing result to ${VTRRRESULT}/${ODO_INPUT}/${ODO_INPUT}"
graph_dir=${VTRRRESULT}/${ODO_INPUT}/${ODO_INPUT}/graph
if [ -d $graph_dir ]; then
  # Count the number of directories inside "graph"
  dir_count=$(ls -l $graph_dir | grep -c ^d)
  
  if [ $dir_count -gt 1 ]; then
    read -p "The directory $graph_dir is not empty and contains $dir_count other directories. Do you want to delete it and create an empty one? (yes/no) " response
    if [ "$response" == "no" ]; then
      # exit
      echo "Running localization on sequence ${LOC_INPUT} to reference sequence ${ODO_INPUT}, storing result to ${VTRRRESULT}/${ODO_INPUT}/${LOC_INPUT}"
    fi
  fi
fi

# ros2 run vtr_testing_radar vtr_testing_radar_radar_odometry \
#   --ros-args -p use_sim_time:=true \
#   -r __ns:=/vtr \
#   --params-file ${PARAM_FILE} \
#   -p data_dir:=${VTRRRESULT}/${ODO_INPUT}/${ODO_INPUT} \
#   -p odo_dir:=${VTRDATA}/${ODO_INPUT}


# # then run localization
# # Log
echo "Running localization on sequence ${LOC_INPUT} to reference sequence ${ODO_INPUT}, storing result to ${VTRRRESULT}/${ODO_INPUT}/${LOC_INPUT}"

# Source the VTR environment with the testing package
source ${VTRROOT}/install/setup.bash

graph_dir=${VTRRRESULT}/${ODO_INPUT}/${LOC_INPUT}/graph
if [ -d $graph_dir ]; then
  # Count the number of directories inside "graph"
  dir_count=$(ls -l $graph_dir | grep -c ^d)
  
  if [ $dir_count -gt 1 ]; then
    read -p "The directory $graph_dir is not empty and contains $dir_count other directories. Do you want to delete it and create an empty one? (yes/no) " response
    if [ "$response" == "no" ]; then
      exit
    fi
  fi
fi

rm -r ${VTRRRESULT}/${ODO_INPUT}/${LOC_INPUT}
mkdir -p ${VTRRRESULT}/${ODO_INPUT}/${LOC_INPUT}
cp -r ${VTRRRESULT}/${ODO_INPUT}/${ODO_INPUT}/* ${VTRRRESULT}/${ODO_INPUT}/${LOC_INPUT}
ros2 run vtr_testing_radar vtr_testing_radar_radar_localization \
  --ros-args -p use_sim_time:=true \
  -r __ns:=/vtr \
  --params-file ${PARAM_FILE} \
  -p data_dir:=${VTRRRESULT}/${ODO_INPUT}/${LOC_INPUT} \
  -p odo_dir:=${VTRDATA}/${ODO_INPUT} \
  -p loc_dir:=${VTRDATA}/${LOC_INPUT}