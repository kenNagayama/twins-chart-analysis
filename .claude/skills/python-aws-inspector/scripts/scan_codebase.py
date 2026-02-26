#!/usr/bin/env python3
"""
Python AWS Inspector - Codebase Scanner
Scans Python files for code quality, security, AWS infrastructure, and performance issues.

Usage:
    python scan_codebase.py [path] [--category CATEGORY] [--output OUTPUT]

Arguments:
    path              Directory or file to scan (default: current directory)
    --category        Filter by category: quality|security|aws|performance|all (default: all)
    --output          Output markdown file path (default: inspection-report.md)
    --severity        Minimum severity to report: low|medium|high (default: low)
"""

import ast
import os
import re
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Issue:
    category: str       # quality | security | aws | performance
    severity: str       # high | medium | low
    file: str
    line: int
    title: str
    description: str
    suggestion: str


def collect_python_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".py" else []
    files = []
    for p in root.rglob("*.py"):
        # skip virtual envs and common non-source dirs
        parts = set(p.parts)
        if parts & {".venv", "venv", "env", "__pycache__", ".git", "node_modules", "dist", "build"}:
            continue
        files.append(p)
    return sorted(files)


# ─── Code Quality Checks ───────────────────────────────────────────────────────

class QualityVisitor(ast.NodeVisitor):
    def __init__(self, source_lines: list[str]):
        self.issues: list[Issue] = []
        self.source_lines = source_lines
        self._file = ""

    def set_file(self, path: str):
        self._file = path

    def visit_FunctionDef(self, node):
        # Long functions (>50 lines)
        end = getattr(node, "end_lineno", node.lineno)
        length = end - node.lineno
        if length > 50:
            self.issues.append(Issue(
                category="quality", severity="medium",
                file=self._file, line=node.lineno,
                title=f"長すぎる関数: {node.name}()（{length} 行）",
                description="50行を超える関数はテストや保守が困難になります。",
                suggestion="より小さく、単一責任の関数に分割してください。",
            ))

        # Missing return type annotation
        if node.returns is None and node.name not in ("__init__", "__str__", "__repr__"):
            self.issues.append(Issue(
                category="quality", severity="low",
                file=self._file, line=node.lineno,
                title=f"戻り値の型アノテーション未記載: {node.name}()",
                description="型アノテーションはIDEサポートを向上させ、型エラーの早期検出に役立ちます。",
                suggestion=f"戻り値型を追加してください: `def {node.name}(...) -> ReturnType:`",
            ))

        # Too many arguments (>5)
        args_count = len(node.args.args)
        if args_count > 5:
            self.issues.append(Issue(
                category="quality", severity="medium",
                file=self._file, line=node.lineno,
                title=f"引数が多すぎる関数: {node.name}()（{args_count} 個）",
                description="引数が多い関数は呼び出しやテストが困難になります。",
                suggestion="関連するパラメータを dataclass や TypedDict にまとめることを検討してください。",
            ))

        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        # Bare except
        if node.type is None:
            self.issues.append(Issue(
                category="quality", severity="high",
                file=self._file, line=node.lineno,
                title="裸の except 節",
                description="SystemExit や KeyboardInterrupt を含むすべての例外を捕捉してしまいます。",
                suggestion="具体的な例外型を指定してください: `except (ValueError, TypeError) as e:`",
            ))
        # Silenced exception (pass only)
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self.issues.append(Issue(
                category="quality", severity="medium",
                file=self._file, line=node.lineno,
                title="例外の握り潰し",
                description="捕捉した例外を無視するとバグの発見が困難になります。",
                suggestion="最低限ログ出力を追加してください: `logger.warning(e)`",
            ))
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        # Class without docstring
        if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)):
            self.issues.append(Issue(
                category="quality", severity="low",
                file=self._file, line=node.lineno,
                title=f"docstring 未記載: class {node.name}",
                description="docstring はクラスの目的と使い方を記録します。",
                suggestion=f'以下を追加してください: `class {node.name}:\n    """概要を記載."""`',
            ))
        self.generic_visit(node)


def check_quality(path: Path, source: str, tree: ast.AST) -> list[Issue]:
    lines = source.splitlines()
    visitor = QualityVisitor(lines)
    visitor.set_file(str(path))
    visitor.visit(tree)

    # Line length check
    for i, line in enumerate(lines, 1):
        if len(line) > 120:
            visitor.issues.append(Issue(
                category="quality", severity="low",
                file=str(path), line=i,
                title=f"行が長すぎる（{len(line)} 文字）",
                description="120文字を超える行は可読性を低下させます。",
                suggestion="行を分割するか、中間変数を使用してください。",
            ))
            break  # report once per file

    return visitor.issues


# ─── Security Checks ───────────────────────────────────────────────────────────

SECURITY_PATTERNS = [
    (r"(password|passwd|secret|api_key|apikey|token)\s*=\s*['\"][^'\"]{4,}['\"]",
     "認証情報のハードコード", "high",
     "ソースコードにハードコードされた認証情報はバージョン管理経由で漏洩する可能性があります。",
     "環境変数または AWS Secrets Manager を使用してください: `os.environ['SECRET']`"),
    (r"subprocess\.(call|run|Popen).*shell\s*=\s*True",
     "シェルインジェクションのリスク", "high",
     "ユーザー入力と組み合わせた `shell=True` はコマンドインジェクションを可能にします。",
     "`shell=True` を避け、引数をリストで渡してください: `subprocess.run(['cmd', arg])`"),
    (r"pickle\.(load|loads|Unpickler)",
     "安全でないデシリアライゼーション (pickle)", "high",
     "信頼できないデータの unpickle は任意コード実行につながります。",
     "信頼できないデータには JSON などの安全なシリアライズ形式を使用してください。"),
    (r"eval\s*\(",
     "eval() の使用", "high",
     "`eval()` は文字列から任意のコードを実行します。",
     "データ解析には `ast.literal_eval()` に置き換えるか、ロジックを再設計してください。"),
    (r"exec\s*\(",
     "exec() の使用", "high",
     "`exec()` は動的に任意のコードを実行します。",
     "動的コード実行を避け、関数マップやストラテジパターンを使用してください。"),
    (r"hashlib\.(md5|sha1)\s*\(",
     "脆弱なハッシュアルゴリズム", "medium",
     "MD5 と SHA-1 は暗号学的に破られています。",
     "SHA-256 以上を使用してください: `hashlib.sha256(data).hexdigest()`"),
    (r"http://",
     "HTTP（非 HTTPS）エンドポイント", "medium",
     "暗号化されていない HTTP は転送中のデータを危険にさらします。",
     "すべてのエンドポイントで HTTPS を使用してください。"),
    (r"DEBUG\s*=\s*True",
     "DEBUG モードが有効", "medium",
     "DEBUG=True はスタックトレースや内部データを露出させる可能性があります。",
     "本番環境では DEBUG=False を確認してください。環境変数で制御することを推奨します。"),
    (r"(os\.system|commands\.getoutput)\s*\(",
     "os.system() の使用", "medium",
     "os.system() はシェルインジェクションに脆弱です。",
     "引数をリストで渡す `subprocess.run()` を使用してください。"),
    (r"logging\.basicConfig.*level\s*=\s*logging\.DEBUG",
     "本番コードに DEBUG ログレベル", "low",
     "DEBUG ログは機密データを漏洩させる可能性があります。",
     "本番環境では INFO または WARNING レベルを使用してください。ログレベルは環境変数で制御してください。"),
]


def check_security(path: Path, source: str) -> list[Issue]:
    issues = []
    lines = source.splitlines()
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern, title, severity, description, suggestion in SECURITY_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append(Issue(
                    category="security", severity=severity,
                    file=str(path), line=i,
                    title=title,
                    description=description,
                    suggestion=suggestion,
                ))
    return issues


# ─── AWS Infrastructure Checks ─────────────────────────────────────────────────

AWS_PATTERNS = [
    # IAM / permissions
    (r'"Effect"\s*:\s*"Allow".*"Action"\s*:\s*"\*"',
     "IAM アクションにワイルドカード", "high",
     "全アクションの許可は最小権限の原則に違反します。",
     "必要なアクションのみ明示的に列挙してください。"),
    (r'"Resource"\s*:\s*"\*"',
     "IAM リソースにワイルドカード", "high",
     "ワイルドカードリソースはそのタイプの全リソースへのアクセスを許可します。",
     "正確なリソース ARN を指定してください。"),
    # S3
    (r'BlockPublicAcls\s*[=:]\s*(False|false|0)',
     "S3 BlockPublicAcls が無効", "high",
     "パブリック ACL によってバケットの内容が公開される可能性があります。",
     "S3 パブリックアクセスブロックの全設定を有効にしてください。"),
    (r'ServerSideEncryption\s*[=:]\s*["\']AES256["\']',
     "S3 が SSE-S3 (AES256) を使用", "low",
     "SSE-S3 は SSE-KMS と比較してキー管理の制御が少ないです。",
     "より細かいキー管理のために SSE-KMS の使用を検討してください。"),
    # Lambda
    (r'Timeout\s*[=:]\s*(\d+)',
     "Lambda タイムアウト設定を検出", "low",
     "タイムアウト値と予想実行時間の関係を確認してください。",
     "最大値 900s ではなく、予想実行時間の 2〜3 倍に設定してください。"),
    (r'MemorySize\s*[=:]\s*128',
     "Lambda が最小メモリ (128 MB) を使用", "low",
     "最小メモリはコールドスタートや実行速度低下を引き起こす可能性があります。",
     "関数をプロファイリングし、パフォーマンス向上のためメモリ増加を検討してください。"),
    # Encryption
    (r'StorageEncrypted\s*[=:]\s*(False|false|0)',
     "RDS StorageEncrypted=False", "high",
     "暗号化されていない RDS ストレージはデータ漏洩のリスクがあります。",
     "すべての RDS インスタンスで StorageEncrypted=True を有効にしてください。"),
    (r'MultiAZ\s*[=:]\s*(False|false|0)',
     "RDS MultiAZ が無効", "medium",
     "シングル AZ の RDS は自動フェイルオーバーができません。",
     "本番データベースでは MultiAZ を有効にしてください。"),
    # Logging
    (r'enable_dns_support\s*[=:]\s*(False|false|0)',
     "VPC DNS サポートが無効", "low",
     "AWS サービスエンドポイントの解決に DNS サポートが必要です。",
     "VPC で DNS サポートを有効にしてください。"),
    (r'deletion_protection\s*[=:]\s*(False|false|0)',
     "削除保護が無効", "medium",
     "削除保護のないリソースは誤って削除される可能性があります。",
     "本番データベースとロードバランサーで deletion_protection を有効にしてください。"),
    # Hardcoded region
    (r'region\s*[=:]\s*["\']us-east-1["\']',
     "AWS リージョンのハードコード", "low",
     "ハードコードされたリージョンはマルチリージョン展開を困難にします。",
     "代わりに `os.environ.get('AWS_REGION', 'us-east-1')` を使用してください。"),
    # boto3 credentials in code
    (r'boto3\.(client|resource)\s*\([^)]*aws_access_key_id\s*=',
     "boto3 呼び出しに明示的な AWS 認証情報", "high",
     "コードに AWS 認証情報を埋め込むと漏洩リスクがあります。",
     "明示的な認証情報を削除してください。IAM ロールまたは環境変数を使用してください。"),
]


def check_aws(path: Path, source: str) -> list[Issue]:
    issues = []
    lines = source.splitlines()
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern, title, severity, description, suggestion in AWS_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE | re.DOTALL):
                issues.append(Issue(
                    category="aws", severity=severity,
                    file=str(path), line=i,
                    title=title,
                    description=description,
                    suggestion=suggestion,
                ))
    return issues


# ─── Performance Checks ────────────────────────────────────────────────────────

class PerformanceVisitor(ast.NodeVisitor):
    def __init__(self):
        self.issues: list[Issue] = []
        self._file = ""
        self._loop_depth = 0

    def set_file(self, path: str):
        self._file = path

    def visit_For(self, node):
        self._loop_depth += 1
        # Nested loops
        if self._loop_depth >= 2:
            self.issues.append(Issue(
                category="performance", severity="medium",
                file=self._file, line=node.lineno,
                title="ネストされたループを検出",
                description="ネストループは O(n²) 以上の計算量になります。",
                suggestion="辞書/セットによるルックアップや numpy/pandas による一括処理を検討してください。",
            ))
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_ListComp(self, node):
        # list comprehension inside loop - skip; handled at call site
        self.generic_visit(node)

    def visit_Call(self, node):
        # Detect `.append()` inside for/while via checking parent context
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "append" and self._loop_depth > 0:
                self.issues.append(Issue(
                    category="performance", severity="low",
                    file=self._file, line=node.lineno,
                    title="ループ内での list.append()",
                    description="繰り返しの append はリスト内包表記で置き換えられます。",
                    suggestion="代わりに `result = [process(x) for x in items]` を使用してください。",
                ))
        self.generic_visit(node)


PERF_PATTERNS = [
    (r'time\.sleep\s*\(\s*[1-9]',
     "1秒以上の time.sleep()", "medium",
     "長いブロッキングスリープはサービスコードのスループットを低下させます。",
     "async/await と asyncio.sleep() を使用するか、SQS/EventBridge による非同期処理を検討してください。"),
    (r'SELECT \*',
     "SQL クエリで SELECT *", "medium",
     "全カラムの選択は不要なデータ転送が発生します。",
     "必要なカラムのみ選択してください: `SELECT id, name FROM ...`"),
    (r'\.objects\.all\(\)',
     "フィルタなしの QuerySet .all()", "medium",
     "全 ORM レコードの取得はメモリを枯渇させる可能性があります。",
     "フィルタ・ページネーションを追加するか、大テーブルには `.values()` / `.values_list()` を使用してください。"),
    (r'global\s+\w+',
     "グローバル変数の使用", "low",
     "グローバル状態はキャッシュや並行処理を困難にします。",
     "状態をクラスにカプセル化するか、関数引数として明示的に渡してください。"),
    (r'json\.loads.*json\.dumps|json\.dumps.*json\.loads',
     "冗長な JSON エンコード/デコード", "low",
     "直後にデシリアライズするシリアライズは CPU を無駄にします。",
     "JSON に変換せず Python オブジェクトを直接渡してください。"),
]


def check_performance(path: Path, source: str, tree: ast.AST) -> list[Issue]:
    visitor = PerformanceVisitor()
    visitor.set_file(str(path))
    visitor.visit(tree)
    issues = list(visitor.issues)

    lines = source.splitlines()
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("#"):
            continue
        for pattern, title, severity, description, suggestion in PERF_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append(Issue(
                    category="performance", severity=severity,
                    file=str(path), line=i,
                    title=title,
                    description=description,
                    suggestion=suggestion,
                ))
    return issues


# ─── Report Generation ─────────────────────────────────────────────────────────

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
CATEGORY_LABELS = {
    "quality": "コード品質",
    "security": "セキュリティ",
    "aws": "AWS インフラ",
    "performance": "パフォーマンス",
}
SEVERITY_LABELS = {"high": "高", "medium": "中", "low": "低"}
SEVERITY_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def generate_report(all_issues: list[Issue], scanned_files: int, target: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Python AWS コードベース調査レポート",
        f"",
        f"- **対象**: `{target}`",
        f"- **日付**: {now}",
        f"- **スキャンファイル数**: {scanned_files}",
        f"- **検出件数合計**: {len(all_issues)}",
        f"",
    ]

    # Summary table
    counts: dict[str, dict[str, int]] = {}
    for issue in all_issues:
        counts.setdefault(issue.category, {"high": 0, "medium": 0, "low": 0})
        counts[issue.category][issue.severity] += 1

    lines += [
        "## サマリー",
        "",
        "| カテゴリ | 🔴 高 | 🟡 中 | 🟢 低 | 合計 |",
        "|---|---|---|---|---|",
    ]
    for cat in ["quality", "security", "aws", "performance"]:
        c = counts.get(cat, {"high": 0, "medium": 0, "low": 0})
        total = sum(c.values())
        lines.append(f"| {CATEGORY_LABELS[cat]} | {c['high']} | {c['medium']} | {c['low']} | {total} |")
    lines.append("")

    # Issues by category
    for cat in ["security", "aws", "quality", "performance"]:
        cat_issues = [i for i in all_issues if i.category == cat]
        if not cat_issues:
            continue
        cat_issues.sort(key=lambda x: SEVERITY_ORDER[x.severity])
        lines += [f"## {CATEGORY_LABELS[cat]}", ""]
        for issue in cat_issues:
            emoji = SEVERITY_EMOJI[issue.severity]
            sev_label = SEVERITY_LABELS[issue.severity]
            lines += [
                f"### {emoji} [{sev_label}] {issue.title}",
                f"",
                f"- **ファイル**: `{issue.file}:{issue.line}`",
                f"- **問題**: {issue.description}",
                f"- **改善策**: {issue.suggestion}",
                "",
            ]

    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scan Python codebase for issues")
    parser.add_argument("path", nargs="?", default=".", help="Directory or file to scan")
    parser.add_argument("--category", default="all",
                        choices=["quality", "security", "aws", "performance", "all"])
    parser.add_argument("--output", default="inspection-report.md")
    parser.add_argument("--severity", default="low", choices=["low", "medium", "high"])
    args = parser.parse_args()

    target = Path(args.path).resolve()
    files = collect_python_files(target)

    if not files:
        print(f"No Python files found in: {target}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(files)} Python file(s)...", file=sys.stderr)

    severity_threshold = SEVERITY_ORDER[args.severity]
    all_issues: list[Issue] = []

    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  [skip] {path}: {e}", file=sys.stderr)
            continue

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            print(f"  [syntax error] {path}: {e}", file=sys.stderr)
            continue

        rel = str(path.relative_to(target.parent if target.is_file() else target))

        if args.category in ("quality", "all"):
            all_issues.extend(check_quality(path, source, tree))
        if args.category in ("security", "all"):
            all_issues.extend(check_security(path, source))
        if args.category in ("aws", "all"):
            all_issues.extend(check_aws(path, source))
        if args.category in ("performance", "all"):
            all_issues.extend(check_performance(path, source, tree))

    # Filter by severity
    all_issues = [i for i in all_issues if SEVERITY_ORDER[i.severity] <= severity_threshold]

    # Update file paths to be relative
    for issue in all_issues:
        try:
            issue.file = str(Path(issue.file).relative_to(target.parent if target.is_file() else target))
        except ValueError:
            pass

    report = generate_report(all_issues, len(files), str(target))

    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to: {output_path}", file=sys.stderr)
    print(f"Issues found: {len(all_issues)}", file=sys.stderr)

    # Also print JSON summary to stdout for programmatic use
    summary = {
        "files_scanned": len(files),
        "total_issues": len(all_issues),
        "by_category": {
            cat: len([i for i in all_issues if i.category == cat])
            for cat in ["quality", "security", "aws", "performance"]
        },
        "report_path": str(output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
