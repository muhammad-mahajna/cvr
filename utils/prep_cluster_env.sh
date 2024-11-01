conda activate FSL
export FSLDIR=/home/muhammad.mahajna/miniconda3/envs/FSL
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:$PATH
export FSLOUTPUTTYPE=NIFTI_GZ