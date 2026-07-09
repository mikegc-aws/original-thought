#!/usr/bin/env bash
# Stop (not terminate) the Phase 3 GPU instance. EBS persists; billing stops (except ~$8/mo disk).
set -euo pipefail
REGION=us-east-1
IID=$(aws ec2 describe-instances --region $REGION \
      --filters "Name=tag:Name,Values=original-thought-gpu" \
                "Name=instance-state-name,Values=pending,running" \
      --query "Reservations[0].Instances[0].InstanceId" --output text)
if [ -z "$IID" ] || [ "$IID" = "None" ]; then
  echo "no running instance"; exit 0
fi
aws ec2 stop-instances --region $REGION --instance-ids $IID >/dev/null
echo "stopping $IID"
