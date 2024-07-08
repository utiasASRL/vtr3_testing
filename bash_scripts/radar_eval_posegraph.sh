# Rebuild vtr_testing_radar
export ASRLROOT=~/ASRL
export ROOTDIR=${ASRLROOT}/vtr3_testing
export VTRROOT=~/ASRL/vtr3         # (INTERNAL default) root directory
export VTRSRC=${VTRROOT}/src       # source code (this repo)
export VTRDATA=${ROOTDIR}/data     # datasets
export VTRTEMP=${VTRROOT}/temp     # default output directory
export VTRMODELS=${VTRROOT}/models # .pt models for TorchScript
export WARTHOG=${VTRROOT}/warthog  # warthog source (this repo)

cd $ROOTDIR
virtualenv venv
source venv/bin/activate

cd $ROOTDIR/pyboreas
pip install -e .

pip install pyyaml
pip install pandas

pip install rosbags
pip install utm

# lets install the posegraph tool
cd ${ASRLROOT}/vtr3_testing/vtr3_pose_graph
pip install -e .

deactivate

echo $VTRDATA
 
source ~/.bashrc
source /opt/ros/humble/setup.bash
source ${VTRSRC}/main/install/setup.bash # source the vtr3 environment
source ${ASRLROOT}/vtr3_testing/test/install/setup.bash # source the vtr_testing_radar environment


export ODO_INPUT=20240708_teach
export LOC_INPUT=20240708_repeat2_forward

# Get arguments
MODE=localization         # [odometry, localization]
SENSOR=radar       # [radar, lidar, radar_lidar]
# ODO_INPUT=$3    # Boreas sequence
# LOC_INPUT=$4    # Boreas sequence, not used if mode=odometry

# Set results subfolder, VTRRESULT is set in setup_container.sh
export VTRRRESULT=${ROOTDIR}/bash_scripts/${SENSOR}


# Load in param file based on sensor
PARAM_FILE=${VTRSRC}/config/navtech_warthog_default.yaml

# Log
echo "Evaluating localization to reference sequence ${ODO_INPUT}, storing result to ${VTRRRESULT}/${ODO_INPUT}"

source ${ROOTDIR}/venv/bin/activate
#   - dump localization result to boreas expected format (txt file)
python ${ASRLROOT}/vtr3_testing/localization_error_eval/eval_loc.py --dataset ${VTRDATA} --path ${VTRRRESULT}/${ODO_INPUT}

# #   - evaluate the result using the evaluation script
# python -m pyboreas.eval.localization --gt ${VTRDATA} --pred ${VTRRRESULT}/${ODO_INPUT}/localization_result --ref_seq ${ODO_INPUT} --ref_sensor radar --test_sensor radar --dim 2  --plot ${VTRRRESULT}/${ODO_INPUT}/localization_result/radar-radar

# # Call corresponding script from vtr_testing_radar
# bash ${ASRLROOT}/vtr3_testing/test/src/vtr_testing_${SENSOR}/script/test_${MODE}_eval.sh ${ODO_INPUT} ${PARAM_FILE}

