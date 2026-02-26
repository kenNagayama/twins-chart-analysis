# AWS Infrastructure Reference

## Well-Architected Framework Quick Checks

### Security Pillar
| Check | Expected |
|---|---|
| IAM least privilege | No wildcard actions/resources |
| S3 Block Public Access | All 4 settings enabled |
| Encryption at rest | RDS, S3, EBS all encrypted |
| Encryption in transit | TLS enforced on all endpoints |
| CloudTrail | Enabled in all regions |
| GuardDuty | Enabled |

### Reliability Pillar
| Check | Expected |
|---|---|
| RDS MultiAZ | True for production |
| Lambda retry | DLQ or destination configured |
| ELB health checks | Properly configured |
| S3 versioning | Enabled for important buckets |
| Backup policy | Automated backups enabled |

### Performance Efficiency Pillar
| Check | Expected |
|---|---|
| Lambda memory | Tuned per function (not default 128MB) |
| Lambda timeout | Set to realistic value, not max 900s |
| RDS instance size | Right-sized (check CloudWatch metrics) |
| CloudFront | Used for static assets |
| ElastiCache | Used for frequent DB reads |

### Cost Optimization Pillar
| Check | Expected |
|---|---|
| Unused resources | No idle EC2, RDS, EIP |
| Lambda reserved concurrency | Set to prevent runaway costs |
| S3 lifecycle rules | Old objects moved to Glacier |
| Savings Plans / Reserved | Used for steady-state workloads |

## Common IaC Patterns (CDK / CloudFormation)

### CDK Security Defaults
```python
# S3 - secure defaults
bucket = s3.Bucket(self, "Bucket",
    block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
    encryption=s3.BucketEncryption.KMS_MANAGED,
    versioned=True,
    removal_policy=RemovalPolicy.RETAIN,
)

# Lambda - least privilege
fn = lambda_.Function(self, "Fn", ...)
# Grant only what's needed:
bucket.grant_read(fn)  # NOT bucket.grant_read_write unless required
```

### Checking Live Resources with AWS CLI

```bash
# List Lambda functions and their timeouts
aws lambda list-functions --query 'Functions[*].[FunctionName,Timeout,MemorySize]' --output table

# Check S3 bucket public access settings
aws s3api get-public-access-block --bucket BUCKET_NAME

# List RDS instances and multi-AZ status
aws rds describe-db-instances --query 'DBInstances[*].[DBInstanceIdentifier,MultiAZ,StorageEncrypted]' --output table

# Check IAM policies for wildcards (requires jq)
aws iam list-policies --scope Local | jq '.Policies[].PolicyName'

# List security groups with 0.0.0.0/0 inbound
aws ec2 describe-security-groups --query "SecurityGroups[?IpPermissions[?IpRanges[?CidrIp=='0.0.0.0/0']]].[GroupId,GroupName]" --output table
```

## Hardcoded Region Anti-pattern

```python
# BAD
client = boto3.client('s3', region_name='us-east-1')

# GOOD
import os
region = os.environ.get('AWS_REGION', 'us-east-1')
client = boto3.client('s3', region_name=region)
```

## boto3 Credentials Best Practice

```python
# BAD - never do this
client = boto3.client('s3',
    aws_access_key_id='AKIA...',
    aws_secret_access_key='...',
)

# GOOD - use IAM roles (ECS task role, Lambda execution role, EC2 instance profile)
client = boto3.client('s3')  # credentials from environment/role automatically
```
