"""Configuration and immutable artifact protocol for this example."""

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
import hashlib
import json
import math
import os
import re
from urllib.parse import parse_qs
from urllib.parse import urlparse

import boto3
from botocore.config import Config as S3Config
import optuna
import yaml

PROTOCOL = 1
TERMINAL = {"COMPLETED", "FAILED", "ERROR", "CANCELLED"}
MAX_FILE = 64 * 1024 * 1024


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def database_url():
    url = os.environ["OPTUNA_STORAGE_URL"]
    parsed = urlparse(url)
    if parsed.scheme != "postgresql+psycopg2":
        raise ValueError("OPTUNA_STORAGE_URL must use postgresql+psycopg2")
    if parsed.hostname not in {"localhost", "127.0.0.1"} and parse_qs(parsed.query).get(
        "sslmode"
    ) != ["verify-full"]:
        raise ValueError("Remote PostgreSQL requires sslmode=verify-full")
    return url


def study_storage(url):
    return optuna.storages.RDBStorage(
        url,
        engine_kwargs={
            "pool_size": 2,
            "max_overflow": 1,
            "pool_timeout": 5,
            "pool_pre_ping": True,
            "connect_args": {"connect_timeout": 10, "options": "-c statement_timeout=15000"},
        },
    )


def pruner():
    return optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)


class ImageReferenceTooLong(ValueError):
    """A new Job would exceed the provider's image-label value limit."""

    def __init__(self):
        super().__init__(
            "Image reference exceeds 128 bytes; shorten the registry/repository path "
            "and retain the full @sha256 digest. Use a new study if changing an existing run."
        )


@dataclass(frozen=True)
class Config:
    project_id: str
    subnet_id: str
    platform: str
    preset: str
    image: str
    region: str
    s3_endpoint: str
    bucket: str
    database_secret_version: str
    s3_secret_version: str
    registry_secret_version: str = ""
    prefix: str = "optuna"
    trials: int = 8
    concurrency: int = 2
    epochs: int = 5
    train_samples: int = 3840
    valid_samples: int = 1280
    seed: int = 42
    job_timeout: int = 3600
    worker_timeout: int = 900
    run_timeout: int = 7200
    poll_seconds: int = 10
    artifact_grace: int = 120
    disk_gib: int = 50
    shm_gib: int = 2

    def validate(self):
        for field in fields(self):
            if field.type is str and not isinstance(getattr(self, field.name), str):
                raise ValueError("Expected a string config field: " + field.name)
        for key, value in asdict(self).items():
            if isinstance(value, str) and ("REPLACE" in value or "<" in value):
                raise ValueError("Replace config placeholders before use")
            if isinstance(value, bool):
                raise ValueError("Boolean config values are not supported")
        if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", self.image):
            raise ValueError("image must include a sha256 digest")
        if not self.preset.startswith("1gpu-"):
            raise ValueError("Select a one-GPU preset (1gpu-...)")
        for key in (
            "project_id",
            "subnet_id",
            "platform",
            "region",
            "bucket",
            "database_secret_version",
            "s3_secret_version",
        ):
            if not getattr(self, key):
                raise ValueError("Missing required config field: " + key)
        endpoint = urlparse(self.s3_endpoint)
        if (
            endpoint.scheme != "https"
            or not endpoint.netloc
            or endpoint.username
            or endpoint.query
        ):
            raise ValueError("S3 endpoint must be an HTTPS URL without credentials/query")
        if not re.fullmatch(r"[a-z0-9][a-z0-9/-]*", self.prefix) or "//" in self.prefix:
            raise ValueError("Use a simple relative S3 prefix")
        bounds = {
            "trials": (1, 100),
            "concurrency": (1, 8),
            "epochs": (1, 100),
            "train_samples": (128, 60000),
            "valid_samples": (128, 10000),
            "seed": (0, 2**31 - 1),
            "job_timeout": (3600, 604800),
            "worker_timeout": (1, 604800),
            "run_timeout": (1, 604800),
            "poll_seconds": (1, 60),
            "artifact_grace": (1, 600),
            "disk_gib": (20, 1024),
            "shm_gib": (1, 64),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid bounded config field: " + name)
        if self.worker_timeout > self.job_timeout or self.concurrency > self.trials:
            raise ValueError("Worker deadline/concurrency exceeds its enclosing bound")
        return self

    def validate_submission(self):
        # Serverless copies the full image into a Compute label, limited to 128 bytes.
        # Keep legacy configs readable for status, reconciliation and cleanup.
        if len(self.image.encode("utf-8")) > 128:
            raise ImageReferenceTooLong()

    @classmethod
    def read(cls, path):
        with open(path) as stream:
            return cls(**yaml.safe_load(stream)).validate()


def prefix_for(config, assignment):
    return (
        f"{config.prefix.rstrip('/')}/{assignment['run_id']}/"
        f"trials/{assignment['trial_number']}/{assignment['submission_id']}/"
    )


def identity(assignment):
    return {
        key: assignment[key]
        for key in ("run_id", "submission_id", "study_name", "trial_number", "params_hash")
    }


class Artifacts:
    def __init__(self, config, client=None, allow_cpu_fixture=False):
        self.config = config
        self.allow_cpu_fixture = allow_cpu_fixture
        self.client = client or boto3.client(
            "s3",
            endpoint_url=config.s3_endpoint,
            region_name=config.region,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            config=S3Config(
                connect_timeout=10,
                read_timeout=20,
                retries={"mode": "standard", "total_max_attempts": 2},
            ),
        )

    def publish(self, assignment, result, files):
        files = {**files, "result.json": encode(result)}
        manifest = {"protocol": PROTOCOL, "identity": identity(assignment), "files": {}}
        for name, content in files.items():
            if not re.fullmatch(r"[a-z][a-z0-9_.-]*", name) or len(content) > MAX_FILE:
                raise ValueError("Invalid output file")
            self.client.put_object(
                Bucket=self.config.bucket,
                Key=prefix_for(self.config, assignment) + name,
                Body=content,
            )
            manifest["files"][name] = {"size": len(content), "sha256": sha256(content)}
        self.client.put_object(
            Bucket=self.config.bucket,
            Key=prefix_for(self.config, assignment) + "manifest.json",
            Body=encode(manifest),
        )
        return sha256(encode(manifest))

    def read(self, assignment, name, limit):
        response = self.client.get_object(
            Bucket=self.config.bucket, Key=prefix_for(self.config, assignment) + name
        )
        body = response["Body"]
        try:
            if response["ContentLength"] > limit:
                raise ValueError("Output exceeds size bound")
            content = body.read(limit + 1)
            if len(content) > limit:
                raise ValueError("Output exceeds size bound")
            return content
        finally:
            body.close()

    def verify(self, assignment):
        raw = self.read(assignment, "manifest.json", 65536)
        manifest = json.loads(raw)
        if manifest.get("protocol") != PROTOCOL or manifest.get("identity") != identity(
            assignment
        ):
            raise ValueError("Manifest identity mismatch")
        if not isinstance(manifest.get("files"), dict) or not 1 <= len(manifest["files"]) <= 8:
            raise ValueError("Invalid manifest entries")
        contents = {}
        for name, spec in manifest["files"].items():
            if name not in {"result.json", "model.pt", "metrics.json", "data.json"}:
                raise ValueError("Unexpected artifact path")
            content = self.read(assignment, name, MAX_FILE)
            if len(content) != spec["size"] or sha256(content) != spec["sha256"]:
                raise ValueError("Artifact integrity mismatch")
            contents[name] = content
        if not {"result.json", "metrics.json", "data.json"} <= contents.keys():
            raise ValueError("Missing required output metadata")
        result = json.loads(contents["result.json"])
        if result.get("identity") != identity(assignment):
            raise ValueError("Result identity mismatch")
        if result.get("state") not in {"COMPLETE", "PRUNED"}:
            raise ValueError("Invalid worker result state")
        value = result.get("value")
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid accuracy")
        epochs = result.get("epochs")
        if type(epochs) is not int or not 1 <= epochs <= self.config.epochs:
            raise ValueError("Invalid completed epoch count")
        if result.get("device") != "cuda" and not (
            self.allow_cpu_fixture and result.get("device") == "cpu-fixture"
        ):
            raise ValueError("Result did not execute on CUDA")
        if result.get("image") != self.config.image or result.get("optuna_version") != "5.0.0":
            raise ValueError("Runtime provenance mismatch")
        if result.get("data_hash") != digest(json.loads(contents["data.json"])):
            raise ValueError("Dataset provenance mismatch")
        metrics = json.loads(contents["metrics.json"])
        if (
            not isinstance(metrics, list)
            or len(metrics) != epochs
            or [m["epoch"] for m in metrics] != list(range(epochs))
            or any(
                type(m["accuracy"]) not in {int, float}
                or not math.isfinite(m["accuracy"])
                or not 0 <= m["accuracy"] <= 1
                for m in metrics
            )
            or metrics[-1]["accuracy"] != value
        ):
            raise ValueError("Metric history mismatch")
        if result["state"] == "COMPLETE":
            if "model.pt" not in contents or epochs != self.config.epochs:
                raise ValueError("Incomplete training result")
        return result, sha256(raw)
