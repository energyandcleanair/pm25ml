---
name: pm25ml-release
description: >-
  Create pm25ml model-bundle Git tags and GitHub releases with the repo's
  versioning scheme and notes format. Use when the user asks to tag, release,
  cut a version, bump the model bundle, or publish mb-X.Y-rT / X.Y-rT.
---

# pm25ml releases

Git tags are `X.Y-rT` (no `mb-` prefix). GitHub release titles are `vX.Y-rT`.
Publish to `energyandcleanair/pm25ml` (`gh --repo energyandcleanair/pm25ml`).
Origin may still be `energyandcleanair/india_map.git`; that remote redirects.

Do not create a tag or GitHub release until the user confirms the version and
notes. Do not bump `pyproject.toml` or add a separate version commit.

## Version number

Model bundle version. Features, ingestion, and model code share one number.
Any of those can require a bump.

Format: `X.Y-rT`

- **X** (major): feature-set or modeling-approach changes.
- **Y** (minor): backward-compatible code changes that keep the feature set identical.
- **T** (retrain): zero-based retrain counter. Resets to `0` when `X.Y` changes.

Read the latest tag from GitHub, not only local tags:

```bash
gh release list --repo energyandcleanair/pm25ml --limit 5
```

Then propose the next version from the commits since that tag:

- Feature set or modeling approach changed → `X+1.0-r0`
- Backward-compatible code/ingestion change, same features → `X.(Y+1)-r0`
- Retrain only, same `X.Y` → `X.Y-r(T+1)`

## Notes format

Copy the structure of GitHub release `v1.2-r0`. The GitHub title stays `vX.Y-rT`.
The body starts with its own heading and a short summary, then the three
sections below. Use `*` bullets. Nested bullets are fine for filenames,
columns, or stage names.

```markdown
## pm25ml X.Y-rT

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

**Full changelog:** https://github.com/energyandcleanair/pm25ml/compare/<prev>...X.Y-rT
```

Rules:

- Always include `## pm25ml X.Y-rT`, the summary paragraph, `### What’s changed`,
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
git tag -a X.Y-rT -m "$(cat <<'EOF'
## pm25ml X.Y-rT

…

**Full changelog:** https://github.com/energyandcleanair/pm25ml/compare/<prev>...X.Y-rT
EOF
)"
git push origin X.Y-rT
gh release create X.Y-rT \
  --repo energyandcleanair/pm25ml \
  --title "vX.Y-rT" \
  --notes "$(cat <<'EOF'
## pm25ml X.Y-rT

…

**Full changelog:** https://github.com/energyandcleanair/pm25ml/compare/<prev>...X.Y-rT
EOF
)"
```

Tag the release commit on `main`. `--notes` must be the full `v1.2-r0`-style
body, not a short `### Changes` list. If `gh` targets the wrong fork, keep
`--repo energyandcleanair/pm25ml`.
