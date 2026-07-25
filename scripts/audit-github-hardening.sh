#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FAILURES=0

pass() { printf 'PASS  %s\n' "$1"; }
warn() { printf 'WARN  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; FAILURES=$((FAILURES + 1)); }

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
check_file ".github/workflows/workflow-lint.yml" "Workflow lint present"
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
    REPO="$("$GH_BIN" repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || true)"
    if [[ -z "$REPO" ]]; then
      warn "GitHub CLI available but repo metadata query failed"
    else
      # Open Dependabot alerts.
      alerts="$("$GH_BIN" api "repos/$REPO/dependabot/alerts" \
        --jq '[.[] | select(.state=="open")] | length' 2>/dev/null || echo "?")"
      if [[ "$alerts" == "0" ]]; then
        pass "No open Dependabot alerts"
      elif [[ "$alerts" == "?" ]]; then
        warn "Could not query Dependabot alerts (token may lack the security_events scope)"
      else
        fail "$alerts open Dependabot alert(s)"
      fi

      # Private vulnerability reporting, which SECURITY.md tells reporters to use.
      pvr="$("$GH_BIN" api "repos/$REPO/private-vulnerability-reporting" \
        --jq .enabled 2>/dev/null || echo "?")"
      if [[ "$pvr" == "true" ]]; then
        pass "Private vulnerability reporting enabled"
      elif [[ "$pvr" == "?" ]]; then
        warn "Could not query private vulnerability reporting"
      else
        fail "Private vulnerability reporting disabled but SECURITY.md directs reporters to it"
      fi

      # The live ruleset must still require checks that exist.
      live_checks="$("$GH_BIN" api "repos/$REPO/rulesets" --jq '.[].id' 2>/dev/null | while read -r id; do
        "$GH_BIN" api "repos/$REPO/rulesets/$id" \
          --jq '.rules[]? | select(.type=="required_status_checks") | .parameters.required_status_checks[].context' 2>/dev/null
      done | sort -u)"
      if [[ -z "$live_checks" ]]; then
        warn "No required status checks found on any live ruleset"
      else
        missing=0
        while IFS= read -r ctx; do
          [[ -z "$ctx" ]] && continue
          # Matrix jobs report as "job-name (matrix-value)"; match on the job name.
          base="${ctx%% (*}"
          # A required context must correspond to a job or workflow name we ship.
          if ! grep -rqF "$base" .github/workflows/ 2>/dev/null; then
            fail "Live ruleset requires '$ctx', which no tracked workflow defines"
            missing=1
          fi
        done <<< "$live_checks"
        if [[ "$missing" -eq 0 ]]; then
          pass "Every required status check maps to a tracked workflow"
        fi
      fi
    fi
  else
    warn "GitHub CLI available but not authenticated"
  fi
else
  warn "GitHub CLI not available from this shell; branch protection and repo settings not checked remotely"
fi

if [[ "$FAILURES" -gt 0 ]]; then
  printf '\n%s check(s) failed.\n' "$FAILURES"
  exit 1
fi

printf '\nAll hardening checks passed.\n'
