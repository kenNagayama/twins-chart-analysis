# Performance Reference

## Python Performance Patterns

### Avoid O(n²) — Use Lookups
```python
# BAD - O(n²)
for item in items:
    if item in large_list:   # linear scan each iteration
        ...

# GOOD - O(n)
lookup = set(large_list)
for item in items:
    if item in lookup:
        ...
```

### List Comprehensions over append()
```python
# BAD
result = []
for x in data:
    result.append(transform(x))

# GOOD
result = [transform(x) for x in data]
```

### Lazy Evaluation with Generators
```python
# BAD - loads all into memory
lines = [process(l) for l in open("big.txt")]

# GOOD - processes one at a time
lines = (process(l) for l in open("big.txt"))
```

### Database: Avoid N+1 Queries
```python
# BAD (Django ORM example) - 1 + N queries
orders = Order.objects.all()
for order in orders:
    print(order.customer.name)   # separate query per order

# GOOD - 2 queries total
orders = Order.objects.select_related('customer').all()
```

### Caching with functools
```python
from functools import lru_cache

@lru_cache(maxsize=256)
def expensive_calculation(n: int) -> int:
    ...
```

## AWS Lambda Performance

| Issue | Fix |
|---|---|
| Cold start > 1s | Use Lambda SnapStart (Java) or provisioned concurrency |
| Heavy imports at top | Move rarely-used imports inside function |
| New boto3 client per invocation | Initialize clients at module level (outside handler) |
| Large deployment package | Use Lambda layers for dependencies |
| Synchronous fan-out | Use SNS/SQS for async fan-out |

### Module-level Client Initialization
```python
# BAD - creates new client on every invocation
def handler(event, context):
    s3 = boto3.client('s3')
    ...

# GOOD - reused across warm invocations
s3 = boto3.client('s3')

def handler(event, context):
    s3.get_object(...)
```

## Async Patterns

```python
# BAD - sequential API calls
result_a = await fetch_a()
result_b = await fetch_b()

# GOOD - parallel execution
result_a, result_b = await asyncio.gather(fetch_a(), fetch_b())
```

## Profiling Commands

```bash
# CPU profiling
python -m cProfile -s cumulative script.py | head -20

# Memory profiling
pip install memory-profiler
python -m memory_profiler script.py

# Lambda: use AWS X-Ray for production traces
# Enable tracing: aws lambda update-function-configuration \
#   --function-name my-fn --tracing-config Mode=Active
```

## SELECT * Anti-pattern

```python
# BAD
cursor.execute("SELECT * FROM users WHERE id = %s", (uid,))

# GOOD
cursor.execute("SELECT id, name, email FROM users WHERE id = %s", (uid,))
```
