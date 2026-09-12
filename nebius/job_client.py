"""Minimal Jobs-only SDK transport. Mutations are dispatched once, never replayed."""

from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
import os

from common import digest
from common import encode

from nebius.aio.request_status import RequestStatus
from nebius.aio.service_error import RequestError
from nebius.api.nebius.ai.v1 import CancelJobRequest
from nebius.api.nebius.ai.v1 import CreateJobRequest
from nebius.api.nebius.ai.v1 import DeleteJobRequest
from nebius.api.nebius.ai.v1 import GetJobRequest
from nebius.api.nebius.ai.v1 import JobServiceClient
from nebius.api.nebius.ai.v1 import JobSpec
from nebius.api.nebius.ai.v1 import ListJobsRequest
from nebius.api.nebius.common.v1 import GetOperationRequest
from nebius.api.nebius.common.v1 import ResourceMetadata
from nebius.api.nebius.compute.v1 import DiskSpec
from nebius.sdk import SDK


@dataclass(frozen=True)
class OperationObservation:
    resource_id: str
    finished_at: datetime | None
    status: RequestStatus | None


class Rejected(Exception):
    """A definitive admission rejection, without a successful Create response."""


class Missing(Exception):
    pass


def labels(config, assignment):
    return {
        "optuna-run": assignment["run_id"],
        "optuna-submission": assignment["submission_id"],
        "optuna-spec": digest({"config": asdict(config), "assignment": assignment}),
    }


def make_request(config, assignment):
    config_bytes = encode({"config": asdict(config), "assignment": assignment})
    if len(config_bytes) > 65536:
        raise ValueError("Assignment exceeds injected-file limit")
    env = []
    for name, version in [
        ("OPTUNA_STORAGE_URL", config.database_secret_version),
        ("AWS_ACCESS_KEY_ID", config.s3_secret_version),
        ("AWS_SECRET_ACCESS_KEY", config.s3_secret_version),
    ]:
        env.append(
            JobSpec.EnvironmentVariable(
                name=name, mysterybox_secret=JobSpec.MysteryBoxSecretRef(version_id=version)
            )
        )
    spec = JobSpec(
        image=config.image,
        platform=config.platform,
        preset=config.preset,
        subnet_id=config.subnet_id,
        public_ip=False,
        preemptible=False,
        restart_attempts=0,
        timeout=timedelta(seconds=config.job_timeout),
        shm_size_bytes=config.shm_gib * 2**30,
        disk=JobSpec.DiskSpec(
            type=DiskSpec.DiskType.NETWORK_SSD, size_bytes=config.disk_gib * 2**30
        ),
        container_command="python",
        args="/example/worker.py /example/assignment.json",
        environment_variables=env,
        injected_files=[
            JobSpec.FileInjection(container_path="/example/assignment.json", content=config_bytes)
        ],
    )
    if config.registry_secret_version:
        spec.registry_credentials = JobSpec.RegistryCredentials(
            mysterybox_secret_version=config.registry_secret_version
        )
    return CreateJobRequest(
        metadata=ResourceMetadata(
            parent_id=config.project_id,
            name="optuna-" + assignment["submission_id"],
            labels=labels(config, assignment),
        ),
        spec=spec,
    )


def visible_spec(spec):
    # Compare fields available without SECRET view. Never request sensitive file contents.
    return {
        "image": spec.image,
        "platform": spec.platform,
        "preset": spec.preset,
        "subnet": spec.subnet_id,
        "public_ip": spec.public_ip,
        "preemptible": spec.preemptible,
        "restarts": spec.restart_attempts,
        "timeout": spec.timeout,
        "disk": (spec.disk.type, spec.disk.size_bytes),
        "shm": spec.shm_size_bytes,
        "working_dir": spec.working_dir,
        "command": spec.container_command,
        "args": spec.args,
        "files": sorted(f.container_path for f in spec.injected_files),
        "env": sorted(
            (e.name, e.value, e.mysterybox_secret.version_id) for e in spec.environment_variables
        ),
        "registry": spec.registry_credentials.mysterybox_secret_version,
        "volumes": len(spec.volumes),
        "ports": len(spec.ports),
        "ssh": list(spec.ssh_authorized_keys),
    }


def matches(job, config, row):
    assignment = row["assignment"]
    expected = make_request(config, assignment)
    metadata = job.metadata
    return (
        metadata.parent_id == config.project_id
        and metadata.name == expected.metadata.name
        and all(metadata.labels.get(k) == v for k, v in labels(config, assignment).items())
        and metadata.created_at is not None
        and metadata.created_at >= row["created_at"] - timedelta(minutes=1)
        and visible_spec(job.spec) == visible_spec(expected.spec)
    )


class Jobs:
    def __init__(self, sdk=None):
        # Explicit credential file avoids accidentally inheriting a different task's profile.
        self.sdk = sdk or SDK(
            credentials_file_name=os.environ["NEBIUS_CREDENTIALS_FILE"],
            user_agent_prefix="optuna-serverless-example/1",
        )
        self.client = JobServiceClient(self.sdk)

    async def close(self):
        await self.sdk.close()

    async def create(self, config, assignment):
        request = make_request(config, assignment)
        try:
            operation = await self.client.create(
                request,
                retries=0,
                timeout=30,
                metadata=[("x-idempotency-key", assignment["submission_id"])],
            )
        except RequestError as exc:
            code = exc.status.code.name
            if code in {"INVALID_ARGUMENT", "PERMISSION_DENIED", "UNAUTHENTICATED"}:
                raise Rejected(code) from None
            raise
        return {"operation_id": operation.id, "job_id": operation.resource_id or None}

    async def get(self, job_id):
        try:
            return await self.client.get(GetJobRequest(id=job_id), retries=2, timeout=20)
        except RequestError as exc:
            if exc.status.code.name == "NOT_FOUND":
                raise Missing() from None
            raise

    async def operation(self, operation_id):
        operation = await self.client.operation_service().get(
            GetOperationRequest(id=operation_id), retries=2, timeout=20
        )
        return OperationObservation(
            operation.resource_id, operation.finished_at, operation.status()
        )

    async def candidates(self, config, row):
        found, token, seen = [], "", set()
        while True:
            page = await self.client.list(
                ListJobsRequest(parent_id=config.project_id, page_size=100, page_token=token),
                retries=2,
                timeout=20,
            )
            # List can be a summary view; Get each matching identity before comparing its spec.
            for job in page.items:
                if all(
                    job.metadata.labels.get(k) == v
                    for k, v in labels(config, row["assignment"]).items()
                ):
                    candidate = await self.get(job.metadata.id)
                    found.append(candidate)
            token = page.next_page_token
            if not token:
                return found
            if token in seen:
                raise RuntimeError("Repeated pagination token; reconciliation incomplete")
            seen.add(token)

    async def cancel(self, job_id):
        operation = await self.client.cancel(CancelJobRequest(id=job_id), retries=0, timeout=30)
        return operation.id

    async def delete(self, job_id):
        operation = await self.client.delete(DeleteJobRequest(id=job_id), retries=0, timeout=30)
        return operation.id
