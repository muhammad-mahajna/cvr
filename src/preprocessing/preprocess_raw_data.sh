RAW_DATA_BASE_DIR="../../../../../network_drive"
RSBOLD_DIR_ET="../../../../data/cvr/rsBOLD_ET"
RSBOLD_DIR_ET_FLIP="../../../../data/cvr/rsBOLD_ET_Flip"
T1_DIR="../../../../data/cvr/T1"
OUTPUT_BASE_DIR=$PWD

./preprocess_fmri_anatomical.sh $RSBOLD_DIR_ET $T1_DIR $OUTPUT_BASE_DIR 19570-20220112-SF01029
#./batch_preprocess_fmri_anat.sh /path/to/rsBOLD_ET /path/to/T1 /path/to/output all
