# GitHub Ruleset Payloads

Repository: `thequantumfalcon/astra-synthetic-gating-demo`

## Apply with GitHub CLI

Create the main branch ruleset:

```bash
gh api --method POST \
  -H "Accept: application/vnd.github+json" \
  /repos/thequantumfalcon/astra-synthetic-gating-demo/rulesets \
  --input .github/rulesets/main-protection.json
```

Create the release tag ruleset:

```bash
gh api --method POST \
  -H "Accept: application/vnd.github+json" \
  /repos/thequantumfalcon/astra-synthetic-gating-demo/rulesets \
  --input .github/rulesets/tag-protection.json
```

List current rulesets:

```bash
gh api /repos/thequantumfalcon/astra-synthetic-gating-demo/rulesets
```

## Apply with Terraform

Use `ops/github-rulesets.tf` with the GitHub provider and import or apply it against `thequantumfalcon/astra-synthetic-gating-demo`.

## Solo-maintainer mode

The checked-in `main-protection` payload currently reflects a solo-maintainer workflow: pull requests are still required, but code-owner review, last-push approval, and the required approval count are disabled so a single maintainer can merge after status checks pass.

## Additional repository settings outside rulesets

These settings should also be enabled in the GitHub UI or through repository settings automation:

- Secret scanning
- Push protection
- Private vulnerability reporting
- Actions policy restricted to GitHub-owned and verified actions
- Default `GITHUB_TOKEN` permissions set to read-only
- First-time contributor workflow approval
