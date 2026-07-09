#!/usr/bin/env bash
# Provision (or restart) the Phase 3 GPU instance. Idempotent.
# Prints the public IP when ready. Companion: gpu_down.sh (stop), gpu_destroy.sh (terminate).
set -euo pipefail

REGION=us-east-1
NAME=original-thought-gpu
TYPE=${GPU_TYPE:-g6.xlarge}          # L4 24GB, ~$0.80/hr on-demand
AMI=ami-012ba162b9cd2729c            # Deep Learning OSS Nvidia PyTorch 2.7 Ubuntu 22.04
KEY_NAME=original-thought-key
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"

# key pair
if ! aws ec2 describe-key-pairs --region $REGION --key-names $KEY_NAME >/dev/null 2>&1; then
  aws ec2 create-key-pair --region $REGION --key-name $KEY_NAME \
    --query KeyMaterial --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  echo "created key $KEY_FILE"
fi

# security group: SSH from current IP only
VPC=$(aws ec2 describe-vpcs --region $REGION --filters Name=is-default,Values=true \
      --query "Vpcs[0].VpcId" --output text)
SG=$(aws ec2 describe-security-groups --region $REGION \
     --filters Name=group-name,Values=$NAME Name=vpc-id,Values=$VPC \
     --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)
if [ -z "$SG" ] || [ "$SG" = "None" ]; then
  SG=$(aws ec2 create-security-group --region $REGION --group-name $NAME \
       --description "original-thought phase3 GPU" --vpc-id $VPC \
       --query GroupId --output text)
fi
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region $REGION --group-id $SG \
  --protocol tcp --port 22 --cidr "${MYIP}/32" 2>/dev/null || true

# instance: reuse stopped one if present
IID=$(aws ec2 describe-instances --region $REGION \
      --filters "Name=tag:Name,Values=$NAME" \
                "Name=instance-state-name,Values=pending,running,stopping,stopped" \
      --query "Reservations[0].Instances[0].InstanceId" --output text 2>/dev/null || true)

if [ -z "$IID" ] || [ "$IID" = "None" ]; then
  IID=$(aws ec2 run-instances --region $REGION --image-id $AMI --instance-type $TYPE \
    --key-name $KEY_NAME --security-group-ids $SG \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=original-thought}]" \
    --query "Instances[0].InstanceId" --output text)
  echo "launched $IID ($TYPE)"
else
  STATE=$(aws ec2 describe-instances --region $REGION --instance-ids $IID \
          --query "Reservations[0].Instances[0].State.Name" --output text)
  if [ "$STATE" = "stopped" ]; then
    aws ec2 start-instances --region $REGION --instance-ids $IID >/dev/null
    echo "starting existing $IID"
  else
    echo "instance $IID already $STATE"
  fi
fi

aws ec2 wait instance-running --region $REGION --instance-ids $IID
IP=$(aws ec2 describe-instances --region $REGION --instance-ids $IID \
     --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
echo "READY  ssh -i $KEY_FILE ubuntu@$IP"
echo "$IP" > /tmp/original-thought-gpu-ip
