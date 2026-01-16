#!/usr/bin/env python3
"""Check Clean Architecture boundaries.

This script verifies that:
- src/application/** does not import from src/infrastructure/**
- src/application/** does not import from src/api/**
- src/domain/** does not import from src/infrastructure/** or src/api/** or src/application/**

Exit code 0 if all checks pass, 1 if violations are found.
"""
import re
import sys
from pathlib import Path
from typing import List, Tuple


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory."""
    return list(directory.rglob("*.py"))


def check_imports(file_path: Path, forbidden_patterns: List[Tuple[str, str]]) -> List[str]:
    """Check file for forbidden import patterns.
    
    Args:
        file_path: Path to Python file
        forbidden_patterns: List of (pattern, description) tuples
        
    Returns:
        List of violation messages
    """
    violations = []
    
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"Error reading {file_path}: {e}"]
    
    for line_num, line in enumerate(content.splitlines(), 1):
        # Skip comments
        if line.strip().startswith("#"):
            continue
            
        for pattern, description in forbidden_patterns:
            if re.search(pattern, line):
                violations.append(
                    f"{file_path}:{line_num} - {description}\n"
                    f"  {line.strip()}"
                )
    
    return violations


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    
    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    violations = []
    
    # Check application layer
    application_dir = src_dir / "application"
    if application_dir.exists():
        app_files = find_python_files(application_dir)
        forbidden_for_app = [
            (r"from\s+src\.infrastructure", "Application layer cannot import from infrastructure"),
            (r"import\s+src\.infrastructure", "Application layer cannot import from infrastructure"),
            (r"from\s+src\.api", "Application layer cannot import from api"),
            (r"import\s+src\.api", "Application layer cannot import from api"),
        ]
        
        for file_path in app_files:
            violations.extend(check_imports(file_path, forbidden_for_app))
    
    # Check domain layer
    domain_dir = src_dir / "domain"
    if domain_dir.exists():
        domain_files = find_python_files(domain_dir)
        forbidden_for_domain = [
            (r"from\s+src\.infrastructure", "Domain layer cannot import from infrastructure"),
            (r"import\s+src\.infrastructure", "Domain layer cannot import from infrastructure"),
            (r"from\s+src\.api", "Domain layer cannot import from api"),
            (r"import\s+src\.api", "Domain layer cannot import from api"),
            (r"from\s+src\.application", "Domain layer cannot import from application"),
            (r"import\s+src\.application", "Domain layer cannot import from application"),
        ]
        
        for file_path in domain_files:
            violations.extend(check_imports(file_path, forbidden_for_domain))
    
    # Report results
    if violations:
        print("ERROR: Architecture boundary violations found:\n", file=sys.stderr)
        for violation in violations:
            print(violation, file=sys.stderr)
        print(f"\nTotal violations: {len(violations)}", file=sys.stderr)
        sys.exit(1)
    else:
        print("OK: All architecture boundaries are respected")
        sys.exit(0)


if __name__ == "__main__":
    main()
