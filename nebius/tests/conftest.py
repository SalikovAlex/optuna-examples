import asyncio
from dataclasses import asdict
from datetime import datetime
from datetime import timezone
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import boto3
import moto
import optuna
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Artifacts  # noqa: E402
from common import Config  # noqa: E402
from common import digest  # noqa: E402
from coordinator import Coordinator  # noqa: E402
from job_client import labels  # noqa: E402
from job_client import make_request  # noqa: E402
from job_client import Missing  # noqa: E402
from job_client import Rejected  # noqa: E402
from journal import Journal  # noqa: E402

from nebius.api.nebius.ai.v1 import Job  # noqa: E402
from nebius.api.nebius.ai.v1 import JobStatus  # noqa: E402


class FakeJobs:
    def __init__(self):
        self.jobs, self.operations = {}, {}
        self.creates, self.cancels, self.deletes = 0, 0, 0
        self.mode = "normal"
        self.cancel_completes = True

    async def create(self, config, assignment):
        self.creates += 1
        if self.mode == "rejected":
            raise Rejected("PERMISSION_DENIED")
        if self.mode == "no_response":
            raise TimeoutError()
        request = make_request(config, assignment)
        request.metadata.id = "aijob-" + uuid.uuid4().hex
        request.metadata.created_at = datetime.now(timezone.utc)
        job = Job(
            metadata=request.metadata,
            spec=request.spec,
            status=JobStatus(state=JobStatus.State.RUNNING),
        )
        self.jobs[job.metadata.id] = job
        op_id = str(uuid.uuid4())
        self.operations[op_id] = job.metadata.id
        if self.mode == "lost_response":
            raise TimeoutError()
        return {"operation_id": op_id, "job_id": job.metadata.id}

    async def get(self, job_id):
        if job_id not in self.jobs:
            raise Missing()
        return self.jobs[job_id]

    async def operation(self, op_id):
        return SimpleNamespace(
            resource_id=self.operations[op_id],
            finished_at=datetime.now(timezone.utc),
            status=SimpleNamespace(code=SimpleNamespace(name="OK")),
        )

    async def candidates(self, config, row):
        return [
            job
            for job in self.jobs.values()
            if all(
                job.metadata.labels.get(k) == v
                for k, v in labels(config, row["assignment"]).items()
            )
        ]

    async def cancel(self, job_id):
        self.cancels += 1
        self.jobs[job_id].status.state = (
            JobStatus.State.CANCELLED if self.cancel_completes else JobStatus.State.CANCELLING
        )
        if self.mode == "cancel_response_lost":
            raise TimeoutError()
        op = str(uuid.uuid4())
        self.operations[op] = job_id
        return op

    async def delete(self, job_id):
        self.deletes += 1
        del self.jobs[job_id]
        if self.mode == "delete_response_lost":
            raise TimeoutError()
        op = str(uuid.uuid4())
        self.operations[op] = job_id
        return op


@pytest.fixture
def config():
    return Config(
        project_id="project-test",
        subnet_id="vpcsubnet-test",
        platform="gpu-test",
        preset="1gpu-test",
        image="registry.test/worker@sha256:" + "a" * 64,
        region="us-east-1",
        s3_endpoint="https://s3.example.test",
        bucket="optuna-test",
        database_secret_version="mbsecver-db",
        s3_secret_version="mbsecver-s3",
        trials=3,
        concurrency=2,
        epochs=2,
        train_samples=128,
        valid_samples=128,
    ).validate()


@pytest.fixture
def env(config):
    url = os.environ.get("OPTUNA_TEST_DSN")
    if not url:
        pytest.skip("Set OPTUNA_TEST_DSN to a disposable PostgreSQL database")
    journal = Journal(url)
    journal.initialize()
    study_name = "test-" + uuid.uuid4().hex
    with moto.mock_aws():
        s3 = boto3.client(
            "s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test"
        )
        s3.create_bucket(Bucket=config.bucket)
        store = Artifacts(config, s3, allow_cpu_fixture=True)
        with journal.lock(study_name):
            run = journal.create_run(study_name, asdict(config), digest(asdict(config)))
            jobs = FakeJobs()
            coord = Coordinator(journal, run, jobs, store, url)
            value = SimpleNamespace(
                config=config,
                journal=journal,
                run=run,
                jobs=jobs,
                store=store,
                coord=coord,
                url=url,
                s3=s3,
            )
            try:
                yield value
            finally:
                optuna.delete_study(study_name=study_name, storage=coord.storage)
                journal.query(
                    "DELETE FROM nebius_optuna.submissions WHERE run_id=%s", (run["run_id"],)
                )
                journal.query("DELETE FROM nebius_optuna.runs WHERE run_id=%s", (run["run_id"],))
        journal.close()


def allocate(env):
    asyncio.run(env.coord.allocate())
    return env.journal.rows(env.run["run_id"])[-1]
