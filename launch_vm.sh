#!/bin/bash

# Set the project IDs
PROJECT_IDS=("pokerai-417521" "idl-hw-3-449021")

# Set the instance configuration
INSTANCE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
DISK_SIZE="70GB"
DISK_TYPE="pd-ssd"
IMAGE_PROJECT="deeplearning-platform-release"
IMAGE_FAMILY="common-cu124-debian-11"
SPOT_INSTANCE=true

# Add commit ID as a parameter
if [ $# -eq 0 ]; then
    echo "Usage: $0 <commit-id>"
    exit 1
fi
COMMIT_ID=$1

list_regions_with_gpus() {
    # Get all regions with T4 GPUs and sort them in the desired priority
    gcloud compute accelerator-types list --filter="name=$GPU_TYPE" --format="value(zone)" | awk -F'-' '{print $1"-"$2}' | sort | uniq | \
    awk '
        /^us/ {print $0 > "us.txt"}
        /^northamerica/ {print $0 > "northamerica.txt"}
        /^southamerica/ {print $0 > "southamerica.txt"}
        /^europe/ {print $0 > "europe.txt"}
        END {
            system("cat us.txt northamerica.txt southamerica.txt europe.txt 2>/dev/null")
            system("rm -f us.txt northamerica.txt southamerica.txt europe.txt 2>/dev/null")
        }
    '
}

# Function to check GPU quota in a region
check_gpu_quota() {
    local region=$1
    local project_id=$2

    echo "Checking GPU quota in region $region for project $project_id..."

    # Get the available GPU quota and usage for the region
    quota_info=$(gcloud compute regions describe $region --project=$project_id --format=json | jq -r '.quotas[] | select(.metric == "PREEMPTIBLE_NVIDIA_T4_GPUS") | "\(.limit) \(.usage)"')
    
    # Extract limit and usage
    quota_limit=$(echo $quota_info | awk '{print $1}')
    quota_usage=$(echo $quota_info | awk '{print $2}')

    # Convert float to integer for comparison
    quota_limit=${quota_limit%.*}
    quota_usage=${quota_usage%.*}

    # Calculate available quota
    available_quota=$((quota_limit - quota_usage))

    if [[ -z "$quota_limit" || "$available_quota" -lt "$GPU_COUNT" ]]; then
        echo "Insufficient GPU quota in region $region. Limit: $quota_limit, Used: $quota_usage, Available: $available_quota"
        return 1
    else
        echo "Sufficient GPU quota available in region $region. Limit: $quota_limit, Used: $quota_usage, Available: $available_quota"
        return 0
    fi
}

# Function to launch the instance
launch_instance() {
    local project_id=$1
    local region=$2

    echo "Attempting to launch instance in project $project_id, region $region..."

    # Try to launch the instance with startup script and commit ID metadata
    gcloud compute instances create "deeplearning-vm-$(date +%s)" \
        --project=$project_id \
        --zone="$region-b" \
        --machine-type=$INSTANCE_TYPE \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --maintenance-policy=TERMINATE \
        --image-project=$IMAGE_PROJECT \
        --image-family=$IMAGE_FAMILY \
        --boot-disk-size=$DISK_SIZE \
        --boot-disk-type=$DISK_TYPE \
        --metadata="install-nvidia-driver=True,commit-id=$COMMIT_ID" \
        --metadata-from-file=startup-script=startup_script.sh \
        --preemptible

    if [[ $? -eq 0 ]]; then
        echo "Instance successfully launched in region $region."
        return 0
    else
        echo "Failed to launch instance in region $region. Trying another region..."
        return 1
    fi
}

# Main script
VM_LAUNCHED=false

for project_id in "${PROJECT_IDS[@]}"; do
    echo "Processing project: $project_id"

    # Set the current project
    gcloud config set project $project_id

    # Get the list of regions with T4 GPUs
    regions=$(list_regions_with_gpus)

    if [[ -z "$regions" ]]; then
        echo "No regions found with available T4 GPUs."
        continue
    fi

    # Try to launch the instance in each region until successful
    for region in $regions; do
        echo "Checking region: $region"

        # Check GPU quota in the region
        if ! check_gpu_quota $region $project_id; then
            continue
        fi

        # Attempt to launch the instance
        if launch_instance $project_id $region; then
            VM_LAUNCHED=true
            break 2  # Break out of both loops
        fi
    done
done

if $VM_LAUNCHED; then
    echo "Instance successfully launched."
else
    echo "Failed to launch instance in any project or region."
    exit 1
fi
