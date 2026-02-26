# Code Quality Reference

## Key Rules

### Functions
- Max 50 lines per function; split if exceeded
- Max 5 parameters; use dataclass/TypedDict for more
- Always add return type annotation (except `__init__`)
- Add docstring for public functions and classes

### Error Handling
- Never use bare `except:` — always name the exception
- Never silently swallow exceptions with `pass`
- Use `logging.exception(e)` to preserve stack traces

### Naming & Structure
- Follow PEP 8: `snake_case` for functions/vars, `PascalCase` for classes
- Max line length: 120 characters
- Use `pathlib.Path` instead of `os.path` string manipulation
- Prefer `f-strings` over `.format()` or `%`

### Type Safety
- Add type hints to all function signatures
- Use `Optional[T]` or `T | None` for nullable values
- Run `mypy` or `pyright` in CI

## Quick Checks

```bash
# Linting
ruff check .

# Type checking
mypy . --ignore-missing-imports

# Complexity
radon cc . -s -n C   # show functions with grade C or worse
```

## Common Refactoring Patterns

| Smell | Fix |
|---|---|
| `if x == True:` | `if x:` |
| `len(lst) == 0` | `not lst` |
| `dict.has_key(k)` | `k in dict` |
| Magic numbers | Named constants / Enum |
| Repeated string literals | Constants at module level |
