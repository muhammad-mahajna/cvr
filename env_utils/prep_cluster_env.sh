conda activate FSL
export FSLDIR=/home/muhammad.mahajna/miniconda3/envs/FSL
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:$PATH
export FSLOUTPUTTYPE=NIFTI_GZ
export PATH="/home/muhammad.mahajna/software/ants-2.5.3/bin:$PATH"