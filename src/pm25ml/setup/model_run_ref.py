"""
Allocate and validate the model-run filename suffix.

The published field meanings live in ``docs/versioning.md``. This module only
formats, parses, and persists those fields for a pipeline execution.
"""

from __future__ import annotations

import re
import secrets
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import arrow
from gcsfs import GCSFileSystem

from pm25ml.logging import logger
from pm25ml.results.netcdf_final_result_writer import SCHEMA_VERSION

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence

    from fsspec import AbstractFileSystem

VERSIONING_DOC = "docs/versioning.md"
RESERVATION_FILENAME = ".reserved"
BUNDLE_SOURCE_RELEASE = "release"
BUNDLE_SOURCE_COMMIT = "commit"
BUNDLE_SOURCE_LOCAL = "local"

_PRODUCTION_REL = re.compile(r"^rel-(?P<year>\d{4})-(?P<month>\d{2})-r(?P<counter>\d+)$")
_LOCAL_REL = re.compile(r"^rel-[0-9a-f]{8}$")
_PRODUCTION_BUNDLE = re.compile(r"^mb-\d+\.\d+\.\d+$")
_LOCAL_BUNDLE = re.compile(r"^mb-local-[0-9a-f]+(?:-dirty)?$")
_LEGACY_BUNDLE = re.compile(r"^mb-\d+\.\d+-r\d+$")
_MODEL_RUN_REF = re.compile(
    r"^(?P<rel>rel-(?:\d{4}-\d{2}-r\d+|[0-9a-f]{8}))"
    r"_s-(?P<schema>\d+\.\d+)"
    r"_(?P<bundle>mb-.+)$",
)


class ModelRunRefError(ValueError):
    """Raised when a model-run reference cannot be allocated or reused."""


@dataclass(frozen=True)
class CombinedRunStorage:
    """Combined-bucket location used to claim and resume production runs."""

    filesystem: AbstractFileSystem
    bucket: str
    profile_id: str


@dataclass(frozen=True)
class ProcessVersions:
    """Schema and model-bundle identity of the running process."""

    schema: str
    bundle: str


@dataclass(frozen=True)
class ParsedModelRunRef:
    """The three filename fields encoded by ``MODEL_RUN_REF``."""

    rel: str
    schema: str
    bundle: str

    def format(self) -> str:
        """Return the filename suffix."""
        return f"{self.rel}_s-{self.schema}_{self.bundle}"


def parse_model_run_ref(value: str) -> ParsedModelRunRef:
    """Parse a run reference or raise if it does not match the filename suffix."""
    cleaned = value.strip()
    match = _MODEL_RUN_REF.fullmatch(cleaned)
    if match is None:
        msg = f"Invalid MODEL_RUN_REF {cleaned!r}. See {VERSIONING_DOC} for the filename schema."
        raise ModelRunRefError(msg)

    parsed = ParsedModelRunRef(
        rel=match.group("rel"),
        schema=match.group("schema"),
        bundle=match.group("bundle"),
    )
    _validate_bundle_token(parsed.bundle)
    return parsed


def _validate_bundle_token(bundle: str) -> None:
    if _LEGACY_BUNDLE.fullmatch(bundle):
        msg = f"MODEL_RUN_REF uses the retired bundle form {bundle!r}. See {VERSIONING_DOC}."
        raise ModelRunRefError(msg)
    if _PRODUCTION_BUNDLE.fullmatch(bundle) or _LOCAL_BUNDLE.fullmatch(bundle):
        return
    msg = f"Invalid model bundle {bundle!r}. See {VERSIONING_DOC}."
    raise ModelRunRefError(msg)


def format_model_run_ref(*, rel: str, schema: str, bundle: str) -> str:
    """Join the filename fields into a run reference."""
    return ParsedModelRunRef(rel=rel, schema=schema, bundle=bundle).format()


def current_schema_version() -> str:
    """Return the schema version written by the NetCDF result writer."""
    return SCHEMA_VERSION


def bundle_source(environ: Mapping[str, str]) -> str:
    """Return the process bundle source, defaulting to a local git checkout."""
    value = environ.get("PM25ML_BUNDLE_SOURCE", "").strip().lower()
    if not value:
        return BUNDLE_SOURCE_LOCAL
    if value in {BUNDLE_SOURCE_RELEASE, BUNDLE_SOURCE_COMMIT, BUNDLE_SOURCE_LOCAL}:
        return value
    msg = (
        f"Invalid PM25ML_BUNDLE_SOURCE {value!r}. "
        f"Expected one of: {BUNDLE_SOURCE_RELEASE}, {BUNDLE_SOURCE_COMMIT}, {BUNDLE_SOURCE_LOCAL}."
    )
    raise ModelRunRefError(msg)


def resolve_model_bundle(
    environ: Mapping[str, str],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Resolve ``mb-*`` for this process."""
    source = bundle_source(environ)
    if source == BUNDLE_SOURCE_RELEASE:
        return _release_bundle(environ)
    if source == BUNDLE_SOURCE_COMMIT:
        return _commit_bundle(environ)
    return _local_git_bundle(run=run)


def _release_bundle(environ: Mapping[str, str]) -> str:
    raw = environ.get("MODEL_BUNDLE", "").strip()
    if not raw:
        msg = "Release runs require MODEL_BUNDLE"
        raise ModelRunRefError(msg)
    bundle = raw if raw.startswith("mb-") else f"mb-{raw}"
    if not _PRODUCTION_BUNDLE.fullmatch(bundle):
        msg = f"Invalid MODEL_BUNDLE {raw!r}. See {VERSIONING_DOC}."
        raise ModelRunRefError(msg)
    return bundle


def _commit_bundle(environ: Mapping[str, str]) -> str:
    sha = environ.get("PM25ML_COMMIT_SHA", "").strip().lower()
    if not sha or not re.fullmatch(r"[0-9a-f]+", sha):
        msg = "Commit runs require PM25ML_COMMIT_SHA"
        raise ModelRunRefError(msg)
    return f"mb-local-{sha}"


def _local_git_bundle(
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> str:
    sha_result = run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    sha = sha_result.stdout.strip().lower()
    if sha_result.returncode != 0 or not sha:
        msg = "Local model bundle requires a git checkout"
        raise ModelRunRefError(msg)
    status_result = run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = bool(status_result.stdout.strip())
    suffix = "-dirty" if dirty else ""
    return f"mb-local-{sha}{suffix}"


def new_local_rel(*, token_hex: Callable[[int], str] = secrets.token_hex) -> str:
    """Return a short random release id that cannot consume a production month."""
    return f"rel-{token_hex(4)}"


def reservation_path(storage: CombinedRunStorage, model_run_ref: str) -> str:
    """Return the reservation object for one profile and run."""
    return (
        f"{storage.bucket}/country={storage.profile_id}/run={model_run_ref}/{RESERVATION_FILENAME}"
    )


def allocation_path(bucket: str, allocation_id: str) -> str:
    """Return the object later stages read for a workflow execution."""
    return f"{bucket}/_allocations/{allocation_id}/model_run_ref.txt"


def profile_runs_root(bucket: str, profile_id: str) -> str:
    """Return the hive prefix that holds run folders for a profile."""
    return f"{bucket}/country={profile_id}"


def list_existing_run_refs(storage: CombinedRunStorage) -> tuple[str, ...]:
    """List one level of ``run=`` prefixes for a profile."""
    root = profile_runs_root(storage.bucket, storage.profile_id)
    filesystem = storage.filesystem
    if not filesystem.exists(root):
        return ()
    entries: Sequence[str] = filesystem.ls(root, detail=False)
    found: list[str] = []
    for entry in entries:
        name = str(entry).rstrip("/").rsplit("/", maxsplit=1)[-1]
        prefix = "run="
        if name.startswith(prefix):
            found.append(name.removeprefix(prefix))
    return tuple(found)


def next_production_rel(
    existing_refs: Iterable[str],
    *,
    now: arrow.Arrow | None = None,
) -> str:
    """Return the next ``rel-YYYY-MM-rT`` for the current UTC month."""
    current = (now or arrow.utcnow()).to("UTC")
    year = current.format("YYYY")
    month = current.format("MM")
    highest = -1
    for ref in existing_refs:
        rel = ref.split("_", maxsplit=1)[0]
        match = _PRODUCTION_REL.fullmatch(rel)
        if match is None:
            continue
        if match.group("year") != year or match.group("month") != month:
            continue
        highest = max(highest, int(match.group("counter")))
    return f"rel-{year}-{month}-r{highest + 1}"


def write_if_absent(filesystem: AbstractFileSystem, path: str, contents: str) -> bool:
    """Create ``path`` only when it does not already exist."""
    parent = path.rsplit("/", maxsplit=1)[0]
    filesystem.makedirs(parent, exist_ok=True)
    if filesystem.exists(path):
        return False
    with filesystem.open(path, "wt") as handle:
        handle.write(contents)
    return True


def reservation_exists(storage: CombinedRunStorage, model_run_ref: str) -> bool:
    """Return whether a production run has already been claimed."""
    return bool(storage.filesystem.exists(reservation_path(storage, model_run_ref)))


def claim_production_run(storage: CombinedRunStorage, model_run_ref: str) -> bool:
    """Claim a production run prefix. Return False when another process won."""
    return write_if_absent(
        storage.filesystem,
        reservation_path(storage, model_run_ref),
        "",
    )


def write_allocated_model_run_ref(
    filesystem: AbstractFileSystem,
    bucket: str,
    allocation_id: str,
    model_run_ref: str,
) -> None:
    """Persist the chosen ref so later workflow stages can load it."""
    path = allocation_path(bucket, allocation_id)
    parent = path.rsplit("/", maxsplit=1)[0]
    filesystem.makedirs(parent, exist_ok=True)
    with filesystem.open(path, "wt") as handle:
        handle.write(model_run_ref)


def read_allocated_model_run_ref(
    filesystem: AbstractFileSystem,
    bucket: str,
    allocation_id: str,
) -> str:
    """Read the ref written by ``s000_resolve_model_run_ref`` for this execution."""
    path = allocation_path(bucket, allocation_id)
    if not filesystem.exists(path):
        msg = (
            f"No allocated MODEL_RUN_REF exists for allocation {allocation_id!r}. "
            "Run s000_resolve_model_run_ref first."
        )
        raise ModelRunRefError(msg)
    with filesystem.open(path, "rt") as handle:
        return parse_model_run_ref(handle.read()).format()


def allocate_production_model_run_ref(
    storage: CombinedRunStorage,
    versions: ProcessVersions,
    *,
    now: arrow.Arrow | None = None,
) -> str:
    """Claim the next production regenerate counter for this month."""
    while True:
        existing = list_existing_run_refs(storage)
        rel = next_production_rel(existing, now=now)
        model_run_ref = format_model_run_ref(
            rel=rel,
            schema=versions.schema,
            bundle=versions.bundle,
        )
        if claim_production_run(storage, model_run_ref):
            return model_run_ref


def validate_explicit_model_run_ref(
    value: str,
    *,
    source: str,
    versions: ProcessVersions,
    storage: CombinedRunStorage | None = None,
) -> str:
    """Validate a resume (release) or overwrite (commit/local) reference."""
    parsed = parse_model_run_ref(value)
    if source == BUNDLE_SOURCE_RELEASE:
        return _validate_release_resume(parsed, versions=versions, storage=storage)
    return _validate_local_overwrite(parsed, schema=versions.schema)


def _validate_release_resume(
    parsed: ParsedModelRunRef,
    *,
    versions: ProcessVersions,
    storage: CombinedRunStorage | None,
) -> str:
    if _PRODUCTION_REL.fullmatch(parsed.rel) is None:
        msg = (
            f"Production resume requires rel-YYYY-MM-rT, got {parsed.rel!r}. See {VERSIONING_DOC}."
        )
        raise ModelRunRefError(msg)
    if parsed.schema != versions.schema:
        msg = (
            f"MODEL_RUN_REF schema s-{parsed.schema} does not match this process "
            f"s-{versions.schema}."
        )
        raise ModelRunRefError(msg)
    if parsed.bundle != versions.bundle:
        msg = f"MODEL_RUN_REF bundle {parsed.bundle} does not match this process {versions.bundle}."
        raise ModelRunRefError(msg)
    if storage is None:
        msg = "Production resume requires combined storage"
        raise ModelRunRefError(msg)
    formatted = parsed.format()
    if not reservation_exists(storage, formatted):
        msg = f"No reservation exists for MODEL_RUN_REF {formatted}"
        raise ModelRunRefError(msg)
    return formatted


def _validate_local_overwrite(parsed: ParsedModelRunRef, *, schema: str) -> str:
    if _LOCAL_REL.fullmatch(parsed.rel) is None:
        msg = f"Local and commit overwrites require rel-<8 hex>, got {parsed.rel!r}."
        raise ModelRunRefError(msg)
    if parsed.schema != schema:
        msg = f"MODEL_RUN_REF schema s-{parsed.schema} does not match this process s-{schema}."
        raise ModelRunRefError(msg)
    if _LOCAL_BUNDLE.fullmatch(parsed.bundle) is None:
        msg = f"Local overwrite requires an mb-local bundle, got {parsed.bundle!r}."
        raise ModelRunRefError(msg)
    return parsed.format()


def resolve_model_run_ref(
    environ: Mapping[str, str],
    filesystem: AbstractFileSystem,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    now: arrow.Arrow | None = None,
    token_hex: Callable[[int], str] = secrets.token_hex,
) -> str:
    """Allocate or reuse ``MODEL_RUN_REF`` for this process."""
    source = bundle_source(environ)
    versions = ProcessVersions(
        schema=current_schema_version(),
        bundle=resolve_model_bundle(environ, run=run),
    )
    explicit = environ.get("MODEL_RUN_REF", "").strip()
    bucket = environ.get("COMBINED_BUCKET_NAME", "").strip()
    profile_id = environ.get("PIPELINE_PROFILE", "").strip()
    storage = CombinedRunStorage(filesystem, bucket, profile_id) if bucket and profile_id else None

    if explicit:
        return validate_explicit_model_run_ref(
            explicit,
            source=source,
            versions=versions,
            storage=storage,
        )

    if source == BUNDLE_SOURCE_RELEASE:
        if storage is None:
            msg = "Production allocation requires PIPELINE_PROFILE and COMBINED_BUCKET_NAME"
            raise ModelRunRefError(msg)
        return allocate_production_model_run_ref(storage, versions, now=now)

    rel = new_local_rel(token_hex=token_hex)
    return format_model_run_ref(rel=rel, schema=versions.schema, bundle=versions.bundle)


def persist_resolved_model_run_ref(
    environ: Mapping[str, str],
    filesystem: AbstractFileSystem,
    model_run_ref: str,
) -> None:
    """Write the chosen ref for later workflow stages when an allocation id is set."""
    allocation_id = environ.get("PM25ML_ALLOCATION_ID", "").strip()
    bucket = environ.get("COMBINED_BUCKET_NAME", "").strip()
    if not allocation_id:
        return
    if not bucket:
        msg = "PM25ML_ALLOCATION_ID requires COMBINED_BUCKET_NAME"
        raise ModelRunRefError(msg)
    write_allocated_model_run_ref(filesystem, bucket, allocation_id, model_run_ref)


def load_model_run_ref_for_process(
    environ: Mapping[str, str],
    filesystem: AbstractFileSystem | None = None,
) -> str:
    """Return ``MODEL_RUN_REF``, loading it from the allocation object when needed."""
    existing = environ.get("MODEL_RUN_REF", "").strip()
    if existing:
        return parse_model_run_ref(existing).format()

    allocation_id = environ.get("PM25ML_ALLOCATION_ID", "").strip()
    bucket = environ.get("COMBINED_BUCKET_NAME", "").strip()
    if not allocation_id:
        msg = "Missing required environment variable: MODEL_RUN_REF"
        raise ModelRunRefError(msg)
    if not bucket:
        msg = "Missing required environment variable: COMBINED_BUCKET_NAME"
        raise ModelRunRefError(msg)

    storage = filesystem if filesystem is not None else GCSFileSystem()
    return read_allocated_model_run_ref(storage, bucket, allocation_id)


def format_resolved_model_run_ref(model_run_ref: str) -> str:
    """Format the allocated run ref so an operator can copy it."""
    return "\n".join(
        [
            "Using MODEL_RUN_REF",
            model_run_ref,
            f"Copy this to resume the run. Field meanings: {VERSIONING_DOC}",
        ],
    )


def log_resolved_model_run_ref(model_run_ref: str) -> None:
    """Log the chosen ref so an operator can copy it for a resume."""
    logger.info(
        "Using MODEL_RUN_REF: %s. Filename schema: %s",
        model_run_ref,
        VERSIONING_DOC,
    )


def apply_model_run_ref_to_environ(
    environ: MutableMapping[str, str],
    model_run_ref: str,
) -> None:
    """Export the chosen ref for later stages in this process."""
    environ["MODEL_RUN_REF"] = model_run_ref
