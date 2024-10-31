
# Pipeline for preparing raw data 
# Do that locally since the files are not available on the cluster

echo "Convert rsBOLD images from DICOM to NIFTII format"
./convert_dicom_to_nifti.sh ../../../../../network_drive ../../../../data/cvr/rsBOLD_ET rsBOLD-End-tidal

echo "Convert T1 images from DICOM to NIFTII format"
./convert_dicom_to_nifti.sh ../../../../../network_drive ../../../../data/cvr/T1 SAG_FSPGR_BRAVO

echo "2. Prepare reference CVR maps"

echo "3. Upload rsBOLD images and CVR maps to the cluster"