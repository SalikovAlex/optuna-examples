import asyncio
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime
from datetime import timezone
import uuid

from common import digest
import grpc
from job_client import Jobs
from job_client import make_request
from job_client import matches
from job_client import Rejected
import pytest

from nebius.aio.channel import Channel
from nebius.aio.channel import NoCredentials
from nebius.api.nebius.ai.v1 import Job
from nebius.api.nebius.ai.v1 import JobServiceClient
from nebius.api.nebius.ai.v1 import JobStatus
from nebius.api.nebius.ai.v1 import ListJobsResponse
from nebius.api.nebius.common.v1 import OperationServiceClient
from nebius.base.options import INSECURE
from nebius.base.resolver import Constant


def assignment():
    params = {"width": 32, "dropout": 0.2, "lr": 0.001, "optimizer": "Adam"}
    return {
        "protocol": 1,
        "run_id": str(uuid.uuid4()),
        "submission_id": str(uuid.uuid4()),
        "study_name": "sdk-test",
        "trial_number": 7,
        "params": params,
        "params_hash": digest(params),
    }


async def with_server(handlers, callback, operation_handlers=None):
    server = grpc.aio.server()
    for service, methods in [
        (JobServiceClient, handlers),
        (OperationServiceClient, operation_handlers or {}),
    ]:
        descriptor = service.get_descriptor()
        registry = service.__registry__
        registered = {}
        for method in descriptor.methods:
            if method.name not in methods:
                continue
            request = registry.message_class(method.input_type.full_name)
            response = registry.message_class(method.output_type.full_name)
            registered[method.name] = grpc.unary_unary_rpc_method_handler(
                methods[method.name],
                request_deserializer=request.FromString,
                response_serializer=response.SerializeToString,
            )
        server.add_generic_rpc_handlers(
            (grpc.method_handlers_generic_handler(descriptor.full_name, registered),)
        )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    sdk = Channel(
        user_agent_prefix="optuna-tests/1",
        resolver=Constant(f"127.0.0.1:{port}"),
        options=[(INSECURE, True)],
        credentials=NoCredentials(),
    )
    try:
        await callback(Jobs(sdk))
    finally:
        await sdk.close()
        await server.stop(0)


@pytest.mark.parametrize("code", [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED])
@pytest.mark.parametrize("image_length", [123, 128])
def test_real_sdk_create_no_replay_and_secret_references(config, code, image_length):
    config = replace(config, image="r/" + "x" * (image_length - 74) + "@sha256:" + "a" * 64)
    config.validate_submission()
    a = assignment()
    calls = []

    async def create(request, context):
        calls.append(dict(context.invocation_metadata()))
        assert request.spec.restart_attempts == 0
        assert not request.spec.preemptible
        assert request.spec.timeout.total_seconds() == 3600
        assert request.spec.image == config.image
        assert all(
            not env.value and env.mysterybox_secret.version_id
            for env in request.spec.environment_variables
        )
        assert calls[-1]["x-idempotency-key"] == a["submission_id"]
        assert b"OPTUNA_STORAGE_URL" not in request.spec.injected_files[0].content
        await context.abort(code, "simulated response loss after receiving Create")

    async def run(jobs):
        with pytest.raises(Exception) as error:
            await jobs.create(config, a)
        assert not isinstance(error.value, Rejected)
        assert len(calls) == 1

    asyncio.run(with_server({"Create": create}, run))


def test_real_sdk_admission_rejection(config):
    calls = []

    async def create(request, context):
        calls.append(request)
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid fixture")

    async def run(jobs):
        with pytest.raises(Rejected, match="INVALID_ARGUMENT"):
            await jobs.create(config, assignment())
        assert len(calls) == 1

    asyncio.run(with_server({"Create": create}, run))


def test_real_sdk_paginated_identity_reconciliation(config):
    a = assignment()
    request = make_request(config, a)
    request.metadata.id = "aijob-fixture"
    request.metadata.created_at = datetime.now(timezone.utc)
    # Normal Get masks injected content. Reconciliation must not need SECRET view.
    request.spec.injected_files[0].content = b""
    job = Job(
        metadata=request.metadata,
        spec=request.spec,
        status=JobStatus(state=JobStatus.State.RUNNING),
    )
    row = {"assignment": a, "created_at": datetime.now(timezone.utc)}
    tokens = []

    async def listing(request, context):
        tokens.append(request.page_token)
        if request.page_token == "":
            return ListJobsResponse(next_page_token="page-two")
        return ListJobsResponse(items=[job])

    async def get(request, context):
        assert request.id == job.metadata.id
        assert request.view == 0
        return job

    async def run(jobs):
        found = await jobs.candidates(config, row)
        assert len(found) == 1 and matches(found[0], config, row)
        assert tokens == ["", "page-two"]

    asyncio.run(with_server({"List": listing, "Get": get}, run))


@pytest.mark.parametrize("method", ["Cancel", "Delete"])
def test_real_sdk_cancellation_and_deletion_not_retried(method):
    calls = []

    async def mutation(request, context):
        calls.append(request.id)
        await context.abort(grpc.StatusCode.UNAVAILABLE, "response loss")

    async def run(jobs):
        with pytest.raises(Exception):
            await getattr(jobs, method.lower())("aijob-fixture")
        assert calls == ["aijob-fixture"]

    asyncio.run(with_server({method: mutation}, run))


def test_spec_fingerprint_covers_immutable_config(config):
    a = assignment()
    before = make_request(config, a).metadata.labels["optuna-spec"]
    changed = type(config)(**{**asdict(config), "epochs": config.epochs + 1})
    assert make_request(changed, a).metadata.labels["optuna-spec"] != before


@pytest.mark.parametrize("method", ["Create", "Cancel", "Delete"])
def test_real_sdk_successful_operation_receipts(config, method):
    from nebius.api.nebius.common.v1 import Operation

    calls = []

    async def mutation(request, context):
        calls.append(request)
        return Operation(id="op-fixture", resource_id="aijob-fixture")

    async def run(jobs):
        if method == "Create":
            result = await jobs.create(config, assignment())
            assert result == {"operation_id": "op-fixture", "job_id": "aijob-fixture"}
        else:
            result = await getattr(jobs, method.lower())("aijob-fixture")
            assert result == "op-fixture"
        assert len(calls) == 1

    asyncio.run(with_server({method: mutation}, run))


@pytest.mark.parametrize(
    "done, code",
    [(False, None), (True, grpc.StatusCode.OK), (True, grpc.StatusCode.PERMISSION_DENIED)],
)
def test_real_sdk_operation_observation(done, code):
    from nebius.aio.request_status import RequestStatus
    from nebius.api.nebius.common.v1 import Operation

    async def get(request, context):
        assert request.id == "op-fixture"
        return Operation(
            id="op-fixture",
            resource_id="aijob-fixture",
            finished_at=datetime.now(timezone.utc) if done else None,
            status=RequestStatus(code, "", [], "", "") if done else None,
        )

    async def run(jobs):
        result = await jobs.operation("op-fixture")
        assert result.resource_id == "aijob-fixture"
        assert (result.finished_at is not None) == done
        assert (result.status.code if result.status else None) == code

    asyncio.run(with_server({}, run, operation_handlers={"Get": get}))
