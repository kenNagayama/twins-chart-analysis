---
name: python-aws-inspector
description: >
  Inspects Python codebases deployed on AWS and produces a prioritized improvement report in Markdown.
  Covers four categories: code quality, security vulnerabilities, AWS infrastructure configuration,
  and performance. Use when the user asks to "inspect", "audit", "review", or "report on" a Python
  codebase or specific Python files, especially in an AWS context. Triggers on phrases like
  "inspect codebase", "find improvements", "コードを調査", "改善点をレポート", "audit the code",
  or "review for security/performance issues".
---

# Python AWS Inspector

## Workflow

### 1. Clarify scope (if not already specified)

Ask the user:
- **Target**: specific file, directory, or entire repo?
- **Category filter**: all / quality / security / aws / performance?
- **Output path**: where to save the Markdown report? (default: `inspection-report.md`)

Skip if the user has already provided this information.

### 2. Run the scanner

```bash
python .claude/skills/python-aws-inspector/scripts/scan_codebase.py \
  <PATH> \
  --category <CATEGORY> \
  --output <OUTPUT_PATH>
```

Options:
- `--category all|quality|security|aws|performance` (default: `all`)
- `--severity low|medium|high` — minimum severity to include (default: `low`)
- `--output <file>` — output Markdown path (default: `inspection-report.md`)

### 3. Supplement with manual review

After the script runs, augment the report with analysis the scanner cannot detect:
- **Architecture-level issues** (e.g., tight coupling, missing separation of concerns)
- **Missing tests** for critical business logic
- **AWS CLI checks** for live resource configuration — see [references/aws-infra.md](references/aws-infra.md)
- **Dependency audit**: `pip-audit` or `safety check`

### 4. Prioritize and present findings

Group issues by severity (High → Medium → Low). For each High issue:
1. Explain the risk in one sentence
2. Show the problematic code snippet
3. Provide a concrete fix example

For Medium/Low issues, a summary table is sufficient.

### 5. Save the report

Write the final report to the requested output file. Confirm path to user.

---

## Language

**すべての出力（レポート本文・マニュアルレビュー補足・ユーザーへの回答）は日本語で記述してください。**
All report content, manual review additions, and user-facing messages must be written in **Japanese**.

---

## Reference Files

Load these files when you need detailed guidance:

| File | When to load |
|---|---|
| [references/code-quality.md](references/code-quality.md) | Reviewing code structure, naming, complexity |
| [references/security.md](references/security.md) | Investigating security findings |
| [references/aws-infra.md](references/aws-infra.md) | Reviewing IaC or running AWS CLI checks |
| [references/performance.md](references/performance.md) | Analyzing performance bottlenecks |

---

## Report Format

The scanner produces a Markdown report with this structure:

```
# Python AWS Codebase Inspection Report
- Target / Date / Files scanned / Total issues

## Summary (table: category × severity)

## Security        ← highest priority first
## AWS Infrastructure
## Code Quality
## Performance
```

Each issue entry includes: severity, file:line, problem description, and a concrete suggestion.

---

## Tips

- Run `bandit -r . -ll` alongside the scanner for deeper security analysis
- For AWS CLI checks, ensure the session has read-only IAM permissions (`ReadOnlyAccess` policy)
- If the codebase is large (>500 files), use `--category security` first to focus on critical issues
- Re-run after fixing to confirm issues are resolved
