#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pass() { printf 'PASS  %s\n' "$1"; }
warn() { printf 'WARN  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; }

find_gh() {
  if command -v gh >/dev/null 2>&1; then
    command -v gh
    return 0
  fi

  if command -v gh.exe >/dev/null 2>&1; then
    command -v gh.exe
    return 0
  fi

  local candidate
  for candidate in \
    "/c/Program Files/GitHub CLI/gh.exe" \
    "/mnt/c/Program Files/GitHub CLI/gh.exe" \
    "/c/Users/${USERNAME:-}/AppData/Local/Programs/GitHub CLI/gh.exe"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

check_file() {
  local path="$1"
  local label="$2"
  if [[ -f "$path" ]]; then
    pass "$label"
  else
    fail "$label"
  fi
}

check_no_unpinned_actions() {
  local found=0
  while IFS= read -r file; do
    if grep -nE 'uses:\s+[^@]+@v[0-9]+' "$file" >/dev/null; then
      fail "Workflow uses unpinned action tag in $file"
      found=1
    fi
  done < <(find .github/workflows -maxdepth 1 -type f -name '*.yml' -o -name '*.yaml')
  if [[ "$found" -eq 0 ]]; then
    pass "Workflow actions pinned to non-tag references in tracked templates"
  fi
}

check_file ".github/CODEOWNERS" "CODEOWNERS present"
check_file ".github/dependabot.yml" "Dependabot config present"
check_file ".github/PULL_REQUEST_TEMPLATE.md" "Pull request template present"
check_file ".github/ISSUE_TEMPLATE/bug_report.md" "Bug report template present"
check_file ".github/workflows/codeql.yml" "CodeQL workflow present"
check_file ".github/workflows/dependency-review.yml" "Dependency review workflow present"
check_file ".github/workflows/sbom.yml" "SBOM workflow present"
check_file ".github/workflows/release-attestation.yml" "Release attestation workflow present"
check_file ".github/workflows/scorecards.yml" "Scorecard workflow present"
check_file ".github/workflows/hardened-ci-template.yml" "Hardened CI template present"
check_file ".github/rulesets/main-protection.json" "Main branch ruleset payload present"
check_file ".github/rulesets/tag-protection.json" "Tag ruleset payload present"
check_file "SECURITY.md" "Security policy present"

if grep -q '^\.github/workflows/\*' .github/CODEOWNERS; then
  pass "Workflows are covered by CODEOWNERS"
else
  fail "Workflows are not covered by CODEOWNERS"
fi

if grep -q '^\.github/CODEOWNERS' .github/CODEOWNERS; then
  pass "CODEOWNERS self-protection configured"
else
  fail "CODEOWNERS file is not self-protected"
fi

if grep -q 'branch_protection_rule:' .github/workflows/scorecards.yml; then
  pass "Scorecard workflow listens for branch protection changes"
else
  warn "Scorecard workflow trigger could be strengthened"
fi

check_no_unpinned_actions

if GH_BIN="$(find_gh 2>/dev/null)"; then
  if "$GH_BIN" auth status >/dev/null 2>&1; then
    repo_json="$("$GH_BIN" repo view --json nameWithOwner,defaultBranchRef 2>/dev/null || true)"
    if [[ -n "$repo_json" ]]; then
      pass "GitHub CLI authenticated for remote security checks"
    else
      warn "GitHub CLI available but repo metadata query failed"
    fi
  else
    warn "GitHub CLI available but not authenticated"
  fi
else
  warn "GitHub CLI not available from this shell; branch protection and repo settings not checked remotely"
fi
