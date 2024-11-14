
# Pipeline for preparing raw data - Converting DICOM images to NIFTII
# Do that locally since the files are not available on the cluster
# You can run each one of the following scripts separatly to speed things up


RAW_DATA_BASE_DIR="../../../../../network_drive"    # raw data folder
BASE_DIR="../../../../data/cvr"                     # output folder
RSBOLD_DIR_ET="rsBOLD_ET"                           # fMRI BOLD data folder suffix
RSBOLD_DIR_ET_FLIP="rsBOLD_ET_Flip"                 # fMRI BOLD data folder suffix - FLIP folder
T1_DIR="T1"                                         # MRI anatomical data folder suffix
MBME_CO2_O2_DIR="MBME_CO2_O2"                       # Reference CVR data folder suffix

# Convert DICOM files to NIFTI in parallel

echo "Convert rsBOLD images from DICOM to NIFTII format"
./convert_dicom_to_nifti.sh $RAW_DATA_BASE_DIR $BASE_DIR $RSBOLD_DIR_ET rsBOLD-End-tidal $RSBOLD_DIR_ET &

echo "Convert rsBOLD images (Flip) from DICOM to NIFTII format"
./convert_dicom_to_nifti.sh $RAW_DATA_BASE_DIR $BASE_DIR $RSBOLD_DIR_ET_FLIP rsBOLD-End-tidal-_Flip_PE $RSBOLD_DIR_ET_FLIP &

echo "Convert T1 images from DICOM to NIFTII format"
./convert_dicom_to_nifti.sh $RAW_DATA_BASE_DIR $BASE_DIR $T1_DIR SAG_FSPGR_BRAVO $T1_DIR &

echo "Convert CO2 and O2 reference images from DICOM to NIFTII format"
./convert_dicom_to_nifti.sh $RAW_DATA_BASE_DIR $BASE_DIR $MBME_CO2_O2_DIR Ax_HYPERMEPI-ASL-CO2-O2 $MBME_CO2_O2_DIR &
./convert_dicom_to_nifti.sh $RAW_DATA_BASE_DIR $BASE_DIR $MBME_CO2_O2_DIR Ax_HYPERMEPI-ASL-FLIP-PE-DIRECTION_CO2 $MBME_CO2_O2_DIR &

# Wait for all background jobs to finish
wait

echo "All DICOM to NIFTI conversions are complete."

# Continue with CVR map processing
echo "Prepare reference CVR maps"
cd ../cvr_processing
./process_cvr_maps.sh $RAW_DATA_BASE_DIR $BASE_DIR

# Upload the results to the cluster. All directories and folder structure should be save the same
echo "3. Upload rsBOLD images, T1 imagws and and CVR maps to the cluster"
scp -r $BASE_DIR muhammad.mahajna@arc.ucalgary.ca:/home/muhammad.mahajna/workspace/research/data/cvr
