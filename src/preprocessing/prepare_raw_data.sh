
# Pipeline for preparing raw data - Converting DICOM images to NIFTII
# Do that locally since the files are not available on the cluster
# You can run each one of the following scripts separatly to speed things up

RAW_DATA_BASE_DIR="../../../../../network_drive"
BASE_DIR="../../../../data/cvr"
RSBOLD_DIR_ET="rsBOLD_ET"
RSBOLD_DIR_ET_FLIP="rsBOLD_ET_Flip"
T1_DIR="T1"
MBME_CO2_O2_DIR="MBME_CO2_O2"

echo "Convert rsBOLD images from DICOM to NIFTII format"
./convert_dicom_to_nifti.sh $RAW_DATA_BASE_DIR $BASE_DIR $RSBOLD_DIR_ET rsBOLD-End-tidal $RSBOLD_DIR_ET

echo "Convert rsBOLD images from DICOM to NIFTII format"
./convert_dicom_to_nifti.sh $RAW_DATA_BASE_DIR $BASE_DIR $RSBOLD_DIR_ET_FLIP rsBOLD-End-tidal-_Flip_PE $RSBOLD_DIR_ET_FLIP

echo "Convert T1 images from DICOM to NIFTII format"
./convert_dicom_to_nifti.sh $RAW_DATA_BASE_DIR $BASE_DIR $T1_DIR SAG_FSPGR_BRAVO $T1_DIR

echo "2. Prepare reference CVR maps"
# ./convert_dicom_to_nifti.sh $RAW_DATA_BASE_DIR $BASE_DIR $MBME_CO2_O2_DIR Ax_HYPERMEPI-ASL-CO2-O2 $MBME_CO2_O2_DIR
# ./convert_dicom_to_nifti.sh $RAW_DATA_BASE_DIR $BASE_DIR $MBME_CO2_O2_DIR Ax_HYPERMEPI-ASL-FLIP-PE-DIRECTION_CO2 $MBME_CO2_O2_DIR
cd ../cvr_processing
./process_cvr_maps.sh $RAW_DATA_BASE_DIR $BASE_DIR

echo "3. Upload rsBOLD images, T1 imagws and and CVR maps to the cluster"

scp -r $BASE_DIR muhammad.mahajna@arc.ucalgary.ca:/home/muhammad.mahajna/workspace/research/data/cvr
