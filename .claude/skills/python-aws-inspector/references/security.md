# Security Reference

## Critical Issues (Fix Immediately)

### Hardcoded Credentials
```python
# BAD
API_KEY = "sk-abc123"

# GOOD
import os
API_KEY = os.environ["API_KEY"]
# or use AWS Secrets Manager:
# secret = boto3.client('secretsmanager').get_secret_value(SecretId='my-secret')
```

### Command Injection
```python
# BAD
subprocess.run(f"ls {user_input}", shell=True)

# GOOD
subprocess.run(["ls", user_input])
```

### Unsafe Deserialization
```python
# BAD
data = pickle.loads(user_bytes)

# GOOD
data = json.loads(user_bytes)  # for untrusted input
```

### eval() / exec()
```python
# BAD
result = eval(user_expression)

# GOOD
result = ast.literal_eval(user_expression)  # safe for literals only
```

## Medium Issues

### Weak Hashing
```python
# BAD (for security)
hashlib.md5(data).hexdigest()
hashlib.sha1(data).hexdigest()

# GOOD
hashlib.sha256(data).hexdigest()
# For passwords use bcrypt/argon2, NOT raw hashlib
```

### HTTP vs HTTPS
- All external endpoints must use `https://`
- For internal VPC traffic, HTTPS is still recommended

### DEBUG Mode
```python
# Ensure this is not True in production
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
```

## AWS-Specific Security

| Risk | Check |
|---|---|
| IAM over-privilege | No `Action: "*"` or `Resource: "*"` |
| S3 public access | BlockPublicAcls=True on all buckets |
| Unencrypted storage | Enable encryption on RDS, S3, EBS |
| CloudTrail disabled | Verify CloudTrail is active in all regions |
| Security group 0.0.0.0/0 | Restrict inbound rules to known CIDRs |

## Security Scanning Tools

```bash
# Static analysis
bandit -r . -ll          # report medium and high only

# Dependency vulnerabilities
pip-audit

# Secrets detection
truffleHog --regex --entropy=False .
# or
detect-secrets scan > .secrets.baseline
```
