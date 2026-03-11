# Security Policy

## Reporting a Vulnerability

If you discover a security issue in this repository, do not open a public issue with exploit details.

Report the issue privately through GitHub security advisories if available for this repository. If private reporting is not enabled, open a minimal public issue that requests a private contact path without including sensitive details or proof-of-concept material.

For high-confidence findings affecting workflow integrity, release artifacts, dependency compromise, or credential exposure, include the impacted commit or tag, a minimal proof of impact, and the smallest reproduction needed to validate the report.

## Scope

This repository is a public reproducibility package for a synthetic demonstration. Reports that materially affect code execution, artifact integrity, or dependency safety are in scope.

In scope:

- GitHub Actions workflow security
- Supply chain compromise or malicious dependency introduction
- Secrets exposure in tracked files, history, or release artifacts
- Code execution flaws that can alter generated verification artifacts

Out of scope:

- Hypothetical issues without a reproducible path
- Reports that require private infrastructure not used by this repository
- Social engineering, phishing, or account recovery requests
- Denial-of-service findings against third-party GitHub infrastructure

## Response Expectations

- Initial triage target: 7 days
- Status update target: 14 days
- Fix timeline: depends on severity and reproducibility impact

## Coordinated Disclosure

Please allow time for triage and remediation before public disclosure. If a fix is accepted, release notes should summarize impact and mitigation without reproducing exploit details.

## Disclosure Guidance

Please avoid publishing exploit details until a fix or mitigation has been prepared.
