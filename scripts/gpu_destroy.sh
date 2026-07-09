#!/usr/bin/env bash
# Terminate the Phase 3 GPU instance entirely (EBS deleted).
set -euo pipefail
REGION=us-east-1
IID=$(aws ec2 describe-instances --region $REGION \
      --filters "Name=tag:Name,Values=original-thought-gpu" \
                "Name=instance-state-name,Values=pending,running,stopping,stopped" \
      --query "Reservations[0].Instances[0].InstanceId" --output text)
if [ -z "$IID" ] || [ "$IID" = "None" ]; then
  echo "no instance"; exit 0
fi
aws ec2 terminate-instances --region $REGION --instance-ids $IID >/dev/null
echo "terminating $IID"
