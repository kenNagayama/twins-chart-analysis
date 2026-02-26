# Steering File Format Guide

Reference for writing effective `CLAUDE.md` and sub-steering files.

## CLAUDE.md Structure

```markdown
# [Project Name]

[1-2 sentence description of what this project does and who uses it.]

## Tech Stack

- **Language**: Python 3.11
- **Framework**: FastAPI
- **Database**: PostgreSQL (via SQLAlchemy)
- **Infrastructure**: AWS ECS Fargate + CDK
- **Testing**: pytest

## Architecture

[Brief description of the system design — layers, main components, data flow.]

Key directories:
- `src/` — core business logic
- `app/` — web layer (routes, middleware)
- `lib/` — AWS CDK stack definitions
- `tests/` — unit and integration tests

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest app/tests/

# Run locally
streamlit run app/CIS_App.py

# Build and deploy
npm run build && cdk deploy
```

## Key Patterns

[Non-obvious conventions that Claude should follow in this codebase.]

- Configuration is loaded from `config.yml` via `src/config.py`; never hardcode values
- Authentication uses AWS Cognito; see `src/auth_aws.py`
- All DynamoDB access goes through `src/helpers.py` helper functions

## Domain Glossary

[Terms specific to the project domain that Claude should understand.]

- **MC**: Maintenance Center (保守センター)
- **Trolley line**: Contact wire (架線) for electric railways
- **Wear measurement**: The thickness reduction of the contact wire over time
```

---

## Guidelines for Content Quality

### DO include:
- Development commands (test, run, build, deploy) — Claude forgets these
- Non-obvious file organization rules
- Domain-specific terminology (especially in non-English projects)
- Architecture decisions with brief rationale
- Key dependencies and their versions
- Where to find configuration / environment variables

### DO NOT include:
- Information Claude already knows (e.g., "Python is a programming language")
- Obvious conventions (e.g., "use snake_case for Python")
- Full API documentation — link to external docs instead
- Secrets or credentials

### Tone:
- Factual, not instructional where possible
- "Uses DynamoDB for storage" > "Always use DynamoDB for storage"
- Short sentences, bullet points preferred over paragraphs

---

## Split File Layout (for large projects)

When `CLAUDE.md` would exceed ~150 lines, split into sub-files:

**CLAUDE.md** (overview + imports):
```markdown
# Project Name

[Overview]

@.claude/steering/architecture.md
@.claude/steering/coding-standards.md
@.claude/steering/dev-workflow.md
```

**`.claude/steering/architecture.md`**:
```markdown
## Architecture

[System design, components, data flow]
```

**`.claude/steering/coding-standards.md`**:
```markdown
## Coding Standards

[Conventions, patterns, naming rules]
```

**`.claude/steering/dev-workflow.md`**:
```markdown
## Development Workflow

[Commands, CI/CD, branching, review process]
```

---

## Real Example: Python + AWS Project

```markdown
# Contact Wire Inspection System (CIS)

Railroad contact wire (架線) wear inspection and analysis system.
Streamlit-based web app deployed on AWS ECS Fargate.

## Tech Stack

- **Language**: Python 3.9
- **UI**: Streamlit 1.43
- **Analysis**: Kalman filter (`pykalman`), OpenCV pixel matching
- **Infrastructure**: AWS CDK + ECS Fargate + DynamoDB
- **Auth**: AWS Cognito (via `streamlit-authenticator`)

## Architecture

Multi-page Streamlit app (`app/pages/`) with shared logic in `app/src/`.
Configuration loaded from `app/config.yml` (49KB YAML with trolley line data).

Key modules:
- `src/kalman.py` / `src/kalman_calc.py` — Kalman filter analysis
- `src/similar_pixel.py` — Pixel-based wear detection
- `src/auth_aws.py` — Cognito authentication
- `src/config.py` — YAML config loader
- `src/helpers.py` — DynamoDB read/write helpers

Infrastructure (`lib/`): ECS Cluster → Fargate task → ECR image, port 8501.

## Development Commands

```bash
# Run locally (from app/)
streamlit run CIS_App.py

# Run tests
cd app && pytest tests/

# Build Docker image
docker build -t cis-app ./app

# Deploy AWS CDK (from project root)
npm run build && cdk deploy
```

## Domain Glossary

- **MC**: 保守センター (Maintenance Center)
- **摩耗**: Wire wear / thickness reduction
- **架線**: Contact wire (overhead electric railway wire)
- **類似ピクセル法**: Pixel similarity method for detecting wire edges
```
