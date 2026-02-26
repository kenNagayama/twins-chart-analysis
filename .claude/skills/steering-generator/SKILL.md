---
name: steering-generator
description: >
  Analyzes the current codebase and generates or updates Claude Code steering files (CLAUDE.md and sub-files).
  Steering files provide persistent project context that Claude Code loads automatically on every session.
  Use when the user asks to: "create steering files", "generate CLAUDE.md", "update project context",
  "steeringファイルを作る/更新する", "CLAUDE.mdを生成/更新する", "プロジェクトのコンテキストを記録する",
  or when starting a new project and Claude needs persistent instructions.
---

# Steering Generator

Analyzes a codebase and generates steering files for Claude Code.

## What Are Steering Files?

Claude Code automatically reads `CLAUDE.md` at project root (and parent directories) at the start of every session. Sub-files can be imported with `@path/to/file.md` syntax. Steering files give Claude persistent context about:

- Project purpose and domain
- Tech stack and architecture
- Directory structure and key files
- Coding conventions and patterns
- Development workflow and commands

## Workflow

```
1. Scan the codebase (run scripts/scan_codebase.py)
2. Determine mode: CREATE (no CLAUDE.md) or UPDATE (CLAUDE.md exists)
3. Generate/update files per format in references/steering-format.md
4. Write CLAUDE.md to project root
5. Write sub-files to .claude/steering/ if content is large
```

### Step 1: Scan the Codebase

Run the scan script to extract project metadata:

```bash
python .claude/skills/steering-generator/scripts/scan_codebase.py
```

The script outputs a structured Markdown summary. Use it as the primary input for generating steering content.

### Step 2: Determine Mode

- **CREATE mode**: No `CLAUDE.md` exists at project root → generate from scratch
- **UPDATE mode**: `CLAUDE.md` already exists → read it first, then update only changed sections

### Step 3: Generate Content

Read `references/steering-format.md` for the expected structure and examples.

**Key principles:**
- Be concise — steering files share context with every task
- Prefer facts over instructions (e.g., "Uses DynamoDB for storage" not "Always use DynamoDB")
- Include development commands (test, build, deploy) since Claude forgets these
- Document non-obvious conventions and domain-specific terminology

### Step 4: File Placement

**Single CLAUDE.md** (for small/medium projects):
- Write everything to `CLAUDE.md` at project root
- Keep under 200 lines

**Split files** (for large projects or many topics):
- Write `CLAUDE.md` with `@import` directives
- Put sub-files in `.claude/steering/`

Example split layout:
```
CLAUDE.md                     # Overview + imports
.claude/steering/
  architecture.md             # System design
  coding-standards.md         # Conventions
  dev-workflow.md             # Commands & workflow
```

`CLAUDE.md` imports them like:
```markdown
@.claude/steering/architecture.md
@.claude/steering/coding-standards.md
```

### Step 5: Confirm with User

After generating, show the user:
1. What files were created/updated
2. The content of `CLAUDE.md` (or summary if long)
3. Ask if any sections need adjustments
