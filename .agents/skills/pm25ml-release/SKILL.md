---
name: pm25ml-release
description: >-
  Create pm25ml model-bundle Git tags and GitHub releases with the repo's
  versioning scheme and notes format. Use when the user asks to tag, release,
  cut a version, bump the model bundle, or publish mb-X.Y.Z / X.Y.Z.
---

# pm25ml releases

Git tags are `X.Y.Z` (no `mb-` prefix). GitHub release titles are `vX.Y.Z`.
Publish to `energyandcleanair/pm25ml` (`gh --repo energyandcleanair/pm25ml`).
Origin may still be `energyandcleanair/india_map.git`; that remote redirects.

Do not create a tag or GitHub release until the user confirms the version and
notes. Do not bump `pyproject.toml` or add a separate version commit.

Publishing `X.Y.Z` is what updates `pm25ml-orchestrator-release`. Operators do
not pass `model_bundle` when they execute either workflow. Retraining is a
`rel` change, not a model-bundle bump. See `docs/VERSIONING.md`.

## Version number

Model bundle version. Features, ingestion, and model code share one number.
Any of those can require a bump.

Format: `X.Y.Z`

- **X** (major): feature-set or modelling-approach changes that significantly
  affect methodology or comparability.
- **Y** (minor): backward-compatible improvements that keep the feature set and
  overall approach, but may change results.
- **Z** (patch): bug fixes or maintenance that preserve the intended methodology
  and feature set.

Read the latest tag from GitHub, not only local tags:

```bash
gh release list --repo energyandcleanair/pm25ml --limit 5
```

Then propose the next version from the commits since that tag:

- Feature set or modelling approach changed → `X+1.0.0`
- Backward-compatible improvement, same features and approach → `X.(Y+1).0`
- Bug fix or maintenance, same methodology → `X.Y.(Z+1)`
- Retrain only → do not bump `mb`

Old tags used `X.Y-rT`. Do not parse those as the next `X.Y.Z`.

## Notes format

Copy the structure of GitHub release `v1.2-r0`. The GitHub title stays `vX.Y.Z`.
The body starts with its own heading and a short summary, then the three
sections below. Use `*` bullets. Nested bullets are fine for filenames,
columns, or stage names.

```markdown
## pm25ml X.Y.Z

One or two sentences on what this release is for, in user-facing language.

### What’s changed

* User-visible outcomes. Say what a consumer or operator can now do, not
  how it was implemented.
* Nested detail is allowed:
  * `example-filename_<ref>.nc`

### Implementation and operational changes

* Concrete internals: new args, paths, stages, schemas, tests, config.

### Migration notes

* What callers, scripts, or downstream consumers must change.

**Full changelog:** https://github.com/energyandcleanair/pm25ml/compare/<prev>...X.Y.Z
```

Rules:

- Always include `## pm25ml X.Y.Z`, the summary paragraph, `### What’s changed`,
  and `**Full changelog:**`.
- Include `### Implementation and operational changes` when there are internals
  worth recording. Include `### Migration notes` when behaviour, paths, args, or
  schemas change for callers.
- Do not collapse this into a single `### Changes` list.
- Do not omit the intro paragraph.
- Prefer the apostrophe in `What’s changed` (as on `v1.2-r0`).

## Publish

After the user confirms version and notes:

```bash
git tag -a X.Y.Z -m "$(cat <<'EOF'
## pm25ml X.Y.Z

…

**Full changelog:** https://github.com/energyandcleanair/pm25ml/compare/<prev>...X.Y.Z
EOF
)"
git push origin X.Y.Z
gh release create X.Y.Z \
  --repo energyandcleanair/pm25ml \
  --title "vX.Y.Z" \
  --notes "$(cat <<'EOF'
## pm25ml X.Y.Z

…

**Full changelog:** https://github.com/energyandcleanair/pm25ml/compare/<prev>...X.Y.Z
EOF
)"
```

Tag the release commit on `main`. `--notes` must be the full `v1.2-r0`-style
body, not a short `### Changes` list. If `gh` targets the wrong fork, keep
`--repo energyandcleanair/pm25ml`.
