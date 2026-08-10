#!/bin/bash
# Usage: ./schedule.sh -env_name "ALE/Pong-v5" -seed 999

QUEUE_FILE="job_queue.txt"

if [ $# -eq 0 ]; then
    echo "Usage: $0 [additional arguments for train.py]"
    echo "Example: $0 -env_name \"ALE/Pong-v5\" -seed 999"
    exit 1
fi

JOB="python -u train.py -config_path \"config_files/STORM.yaml\" $@"

# Ensure queue file exists
touch "$QUEUE_FILE"

# Add to the TOP of the queue (since it's a priority schedule)
# Create a temporary file
TMP_FILE="job_queue.tmp"
echo "$JOB" > "$TMP_FILE"
cat "$QUEUE_FILE" >> "$TMP_FILE"
mv "$TMP_FILE" "$QUEUE_FILE"

echo "Scheduled at the top of the queue:"
echo "$JOB"
