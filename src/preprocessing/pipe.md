# Preprocessing pipeline: this pipeline will prepare all data (fMRI+Anatomical+CVR)

1. Make sure that data exists in local folders 
2. run `./prepare_raw_data.sh` locally to create working data and upload this data to the cluster.
3. run `./preproc_all_subs__slurm_job.slurm` in the cluster to perform all preprocessing steps on all available subjects.
4. run `./proc_all_subs_cvr__slurm_job.slurm` in the cluster to perform CVR dta preprocessing steps on all available subjects.
