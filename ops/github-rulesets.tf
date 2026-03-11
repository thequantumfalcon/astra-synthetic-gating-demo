terraform {
  required_providers {
    github = {
      source = "integrations/github"
      version = ">= 6.0"
    }
  }
}

resource "github_repository_ruleset" "main_protection" {
  repository  = "astra-synthetic-gating-demo"
  name        = "main-protection"
  target      = "branch"
  enforcement = "active"

  conditions {
    ref_name {
      include = ["~DEFAULT_BRANCH", "refs/heads/main"]
      exclude = []
    }
  }

  rules {
    deletion         = true
    non_fast_forward = true
    required_linear_history = true

    pull_request {
      dismiss_stale_reviews_on_push     = true
      require_code_owner_review         = false
      require_last_push_approval        = false
      required_approving_review_count   = 0
      required_review_thread_resolution = true
    }

    required_status_checks {
      strict_required_status_checks_policy = true

      required_check {
        context = "reproduce-and-verify (windows-latest)"
      }

      required_check {
        context = "reproduce-and-verify (ubuntu-latest)"
      }

      required_check {
        context = "build-pdf (ubuntu-latest)"
      }

      required_check {
        context = "Analyze Python"
      }

      required_check {
        context = "dependency-review"
      }

      required_check {
        context = "generate-sbom"
      }
    }
  }
}

resource "github_repository_ruleset" "release_tags" {
  repository  = "astra-synthetic-gating-demo"
  name        = "release-tags-protection"
  target      = "tag"
  enforcement = "active"

  conditions {
    ref_name {
      include = ["refs/tags/v*"]
      exclude = []
    }
  }

  rules {
    deletion = true
    update   = true
  }
}
