
# Pipeline for preparing raw data - Converting DICOM images to NIFTII
# Do that locally since the files are not available on the cluster
# You can run each one of the following scripts separatly to speed things up

RAW_DATA_BASE_DIR="../../../../../network_drive"
BASE_DIR="../../../../data/cvr/"
RSBOLD_DIR_ET="rsBOLD_ET"
RSBOLD_DIR_ET_FLIP="rsBOLD_ET_Flip"
T1_DIR="T1"

echo "Convert rsBOLD images from DICOM to NIFTII format"
./convert_dicom_to_nifti.sh $RAW_DATA_BASE_DIR $BASE_DIR $RSBOLD_DIR_ET rsBOLD-End-tidal $RSBOLD_DIR_ET

echo "Convert rsBOLD images from DICOM to NIFTII format"
./convert_dicom_to_nifti.sh $RAW_DATA_BASE_DIR $BASE_DIR $RSBOLD_DIR_ET_FLIP rsBOLD-End-tidal-_Flip_PE $RSBOLD_DIR_ET_FLIP

echo "Convert T1 images from DICOM to NIFTII format"
./convert_dicom_to_nifti.sh $RAW_DATA_BASE_DIR $BASE_DIR $T1_DIR SAG_FSPGR_BRAVO $T1_DIR

echo "2. Prepare reference CVR maps"

echo "3. Upload rsBOLD images and CVR maps to the cluster"