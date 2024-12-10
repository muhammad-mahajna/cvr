# Example usage
# submit_job_and_monitor slurm_job.sh

submit_job_and_monitor() {
    # Submit the job and capture the job ID
    job_id=$(sbatch "$1" | awk '{print $4}')
    
    # Print the submitted Job ID
    echo "Submitted job with Job ID: $job_id"

    # Initialize the state variable
    state=""

    # Loop to check job status until it is RUNNING, COMPLETED, or FAILED
    while true; do
        # Get the current job state using scontrol
        state=$(scontrol show job "$job_id" | grep "JobState=" | awk -F "=" '{print $2}')
        echo "Current job state: $state"
        
        # Break the loop if the job is running, completed, or failed
        if [[ "$state" == "RUNNING" ]] || [[ "$state" == "COMPLETED" ]] || [[ "$state" == "FAILED" ]]; then
            break
        fi

        # Sleep for 5 seconds before checking again
        sleep 5
    done

    # Print the final job state and return the job ID and state
    echo "Job $job_id has finished with state: $state"
}
