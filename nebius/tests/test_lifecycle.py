import asyncio
from dataclasses import asdict
from dataclasses import replace
import io
import json
import math
import os
import subprocess
import sys

from common import Artifacts
from common import Config
from common import database_url
from common import digest
from common import encode
from common import identity
from common import ImageReferenceTooLong
from common import prefix_for
from conftest import allocate
from coordinator import Coordinator
from journal import Journal
from optuna.trial import TrialState
import pytest
import torch
from torch.utils.data import TensorDataset
from worker import model_for
from worker import report
from worker import run_worker

from nebius.api.nebius.ai.v1 import Job
from nebius.api.nebius.ai.v1 import JobStatus
from nebius.api.nebius.common.v1 import ResourceMetadata


class Crash(BaseException):
    pass


def restart(env):
    env.coord = Coordinator(env.journal, env.run, env.jobs, env.store, env.url)
    return env.coord


def complete(env, row, value=0.5):
    assignment = row["assignment"]
    tid = env.coord.storage.get_trial_id_from_study_id_trial_number(
        env.coord.storage.get_study_id_from_name(env.run["study_name"]), row["trial_number"]
    )
    for step in range(env.config.epochs):
        env.coord.storage.set_trial_intermediate_value(tid, step, value)
    data = {"fixture": True}
    result = {
        "identity": identity(assignment),
        "state": "COMPLETE",
        "value": value,
        "epochs": env.config.epochs,
        "device": "cuda",
        "image": env.config.image,
        "optuna_version": "5.0.0",
        "data_hash": digest(data),
    }
    env.store.publish(
        assignment,
        result,
        {
            "model.pt": b"artifact-integrity-test",
            "metrics.json": encode(
                [{"epoch": s, "accuracy": value} for s in range(env.config.epochs)]
            ),
            "data.json": encode(data),
        },
    )
    env.jobs.jobs[row["job_id"]].status.state = JobStatus.State.COMPLETED


@pytest.mark.parametrize(
    "event, creates",
    [
        ("before_ask", 0),
        ("after_ask", 0),
        ("prepared", 0),
        ("before_create", 0),
        ("after_create", 1),
        ("receipt_saved", 1),
    ],
)
def test_crash_boundaries_never_redispatch(env, event, creates):
    def checkpoint(current):
        if current == event:
            raise Crash()

    env.coord.checkpoint = checkpoint
    with pytest.raises(Crash):
        allocate(env)
    restart(env)
    asyncio.run(env.coord.reconcile())
    asyncio.run(env.coord.reconcile())
    assert env.jobs.creates == creates
    row = env.journal.rows(env.run["run_id"])[0]
    if event in {"before_ask", "after_ask", "prepared"}:
        assert row["state"] in {"FINAL", "ABANDONED"}
    elif event == "before_create":
        assert row["state"] == "DISPATCH_INTENT"
    else:
        assert row["job_id"]


def test_lost_create_response_reattaches_without_replay(env):
    env.jobs.mode = "lost_response"
    row = allocate(env)
    assert row["state"] == "SUBMISSION_UNKNOWN"
    assert asyncio.run(restart(env).reconcile())
    recovered = env.journal.row(row["submission_id"])
    assert recovered["job_id"] in env.jobs.jobs
    assert env.jobs.creates == 1


def test_no_candidate_blocks_run_and_keeps_slot(env):
    env.jobs.mode = "no_response"
    allocate(env)
    assert not asyncio.run(restart(env).drive())
    assert len(env.journal.rows(env.run["run_id"])) == 1
    assert env.jobs.creates == 1


def test_duplicate_candidates_refuse_attachment(env):
    env.jobs.mode = "lost_response"
    row = allocate(env)
    original = next(iter(env.jobs.jobs.values()))
    metadata = ResourceMetadata.from_json(original.metadata.to_json())
    metadata.id = "aijob-duplicate"
    env.jobs.jobs[metadata.id] = Job(metadata=metadata, spec=original.spec, status=original.status)
    assert not asyncio.run(restart(env).reconcile())
    assert env.journal.row(row["submission_id"])["job_id"] is None
    assert env.jobs.creates == 1


def test_wrong_spec_refuses_cancellation(env):
    row = allocate(env)
    env.jobs.jobs[row["job_id"]].spec.image = "wrong"
    assert not asyncio.run(env.coord.cancel())
    assert env.jobs.cancels == 0


@pytest.mark.parametrize("state", ["FAILED", "ERROR", "CANCELLED"])
def test_failed_job_not_used_as_objective(env, state):
    row = allocate(env)
    env.jobs.jobs[row["job_id"]].status.state = JobStatus.State[state]
    assert asyncio.run(env.coord.reconcile())
    trial = env.coord.study.trials[0]
    assert trial.state == TrialState.FAIL and trial.value is None
    assert env.coord.summary()["best_trial"] is None


def test_admission_rejection_is_fail_without_replacement(env):
    env.jobs.mode = "rejected"
    row = allocate(env)
    assert row["state"] == "FINAL" and row["job_id"] is None
    assert env.coord.study.trials[0].state == TrialState.FAIL
    assert env.jobs.creates == 1


@pytest.mark.parametrize("event", ["before_tell", "after_tell"])
def test_crash_at_tell_recovers_same_outcome(env, event):
    row = allocate(env)
    complete(env, row)

    def checkpoint(current):
        if current == event:
            raise Crash()

    env.coord.checkpoint = checkpoint
    with pytest.raises(Crash):
        asyncio.run(env.coord.reconcile())
    assert asyncio.run(restart(env).reconcile())
    assert env.coord.study.trials[0].value == 0.5
    assert env.journal.row(row["submission_id"])["state"] == "FINAL"
    assert env.jobs.creates == 1


def test_conflicting_tell_fails_closed(env):
    row = allocate(env)
    complete(env, row)
    env.coord.study.tell(row["trial_number"], 0.1)
    assert not asyncio.run(env.coord.reconcile())
    assert env.coord.study.trials[0].value == 0.1
    with pytest.raises(ValueError):
        asyncio.run(restart(env).reconcile())


def test_corrupt_output_waits_then_fails(env):
    row = allocate(env)
    complete(env, row)
    env.s3.put_object(
        Bucket=env.config.bucket,
        Key=prefix_for(env.config, row["assignment"]) + "model.pt",
        Body=b"corrupt",
    )
    assert not asyncio.run(env.coord.reconcile())
    assert env.coord.study.trials[0].state == TrialState.RUNNING
    env.journal.query(
        "UPDATE nebius_optuna.submissions SET terminal_seen_at="
        "now()-interval '1 hour' WHERE submission_id=%s",
        (row["submission_id"],),
    )
    assert asyncio.run(env.coord.reconcile())
    assert env.coord.study.trials[0].state == TrialState.FAIL


def test_cancel_response_loss_is_not_replayed(env):
    env.jobs.mode = "cancel_response_lost"
    env.jobs.cancel_completes = False
    allocate(env)
    assert not asyncio.run(env.coord.cancel())
    assert not asyncio.run(restart(env).cancel())
    assert env.jobs.cancels == 1


def test_delete_requires_final_result_and_operation_confirmation(env):
    row = allocate(env)
    assert not asyncio.run(env.coord.cleanup())
    assert env.jobs.deletes == 0
    complete(env, row)
    assert asyncio.run(env.coord.reconcile())
    assert not asyncio.run(env.coord.cleanup())
    env.journal.update(row["submission_id"], diagnostic="PreviousReadFailure")
    assert asyncio.run(restart(env).cleanup())
    assert env.journal.row(row["submission_id"])["deleted"]
    assert env.journal.row(row["submission_id"])["diagnostic"] is None
    assert env.jobs.deletes == 1
    # User data is retained after Job metadata deletion.
    assert env.store.verify(row["assignment"])[0]["value"] == 0.5


def test_delete_response_loss_is_unresolved(env):
    row = allocate(env)
    complete(env, row)
    asyncio.run(env.coord.reconcile())
    env.jobs.mode = "delete_response_lost"
    assert not asyncio.run(env.coord.cleanup())
    assert not asyncio.run(restart(env).cleanup())
    assert env.jobs.deletes == 1
    assert not env.journal.row(row["submission_id"])["deleted"]


def test_complete_cancel_race_preserves_completed_result(env):
    row = allocate(env)
    complete(env, row)
    assert asyncio.run(env.coord.cancel())
    assert asyncio.run(env.coord.reconcile())
    assert env.jobs.cancels == 0
    assert env.coord.study.trials[0].state == TrialState.COMPLETE


def test_run_deadline_cancels_existing_jobs_without_new_create(env):
    allocate(env)
    env.journal.query(
        "UPDATE nebius_optuna.runs SET deadline=now()-interval '1 second' " "WHERE run_id=%s",
        (env.run["run_id"],),
    )
    assert not asyncio.run(env.coord.drive())
    assert env.jobs.creates == 1 and env.jobs.cancels == 1


def test_second_coordinator_lock_refused(env):
    other = Journal(env.url)
    try:
        with pytest.raises(RuntimeError):
            with other.lock(env.run["study_name"]):
                pass
    finally:
        other.close()


def test_total_allocation_budget_is_not_success_budget(env):
    env.jobs.mode = "rejected"
    for _ in range(env.config.trials):
        allocate(env)
    with pytest.raises(RuntimeError):
        allocate(env)
    assert env.jobs.creates == env.config.trials


def test_worker_cpu_fixture_roundtrip_and_duplicate_claim(env):
    row = allocate(env)
    generator = torch.Generator().manual_seed(0)
    data = torch.randn(128, 1, 28, 28, generator=generator)
    labels = torch.randint(0, 10, (128,), generator=generator)
    fixture = (TensorDataset(data, labels), TensorDataset(data, labels), {"fixture": True})
    result = run_worker(
        env.config, row["assignment"], env.journal, env.store, env.url, fixture=fixture
    )
    assert result["device"] == "cpu-fixture"
    with pytest.raises(ValueError):
        run_worker(env.config, row["assignment"], env.journal, env.store, env.url, fixture=fixture)
    env.jobs.jobs[row["job_id"]].status.state = JobStatus.State.COMPLETED
    assert asyncio.run(restart(env).reconcile())
    assert env.coord.study.trials[0].state == TrialState.COMPLETE
    raw = env.store.read(row["assignment"], "model.pt", 64 * 1024 * 1024)
    model = model_for(row["assignment"]["params"])
    model.load_state_dict(torch.load(io.BytesIO(raw), weights_only=True))
    model.eval()
    assert torch.isfinite(model(data[:2])).all()
    # Live coordinator must refuse CPU fixture results.
    with pytest.raises(ValueError, match="CUDA"):
        Artifacts(env.config, env.s3).verify(row["assignment"])


def test_worker_pruning_is_optuna_pruned_not_failed(env):
    # Two high-scoring completed baselines meet MedianPruner's startup requirement.
    for _ in range(2):
        baseline = allocate(env)
        complete(env, baseline, value=1.0)
        assert asyncio.run(env.coord.reconcile())
    row = allocate(env)
    data = torch.zeros(128, 1, 28, 28)
    labels = torch.arange(128) % 10
    dataset = TensorDataset(data, labels)
    result = run_worker(
        env.config,
        row["assignment"],
        env.journal,
        env.store,
        env.url,
        fixture=(dataset, dataset, {"fixture": True}),
    )
    assert result["state"] == "PRUNED"
    env.jobs.jobs[row["job_id"]].status.state = JobStatus.State.COMPLETED
    assert asyncio.run(restart(env).reconcile())
    assert env.coord.study.trials[-1].state == TrialState.PRUNED
    assert env.coord.study.best_value == 1.0


def test_invalid_reporting_never_overwrites_epoch(env):
    row = allocate(env)
    storage, study = env.coord.storage, env.coord.study
    tid = storage.get_trial_id_from_study_id_trial_number(
        storage.get_study_id_from_name(study.study_name), row["trial_number"]
    )
    report(storage, study, tid, 0, 0.5)
    for step, value in [(0, 0.1), (2, 0.5), (1, math.nan), (-1, 0.5)]:
        with pytest.raises(ValueError):
            report(storage, study, tid, step, value)
    assert storage.get_trial(tid).intermediate_values == {0: 0.5}


@pytest.mark.parametrize(
    "field,value",
    [
        ("trials", 0),
        ("concurrency", 100),
        ("image", "worker:latest"),
        ("job_timeout", 60),
        ("preset", "8gpu-test"),
        ("prefix", "../other"),
    ],
)
def test_config_rejects_unsafe_or_unbounded_values(config, field, value):
    values = asdict(config)
    values[field] = value
    with pytest.raises(ValueError):
        Config(**values).validate()


def test_remote_database_requires_real_verify_full_query(monkeypatch):
    monkeypatch.setenv(
        "OPTUNA_STORAGE_URL", "postgresql+psycopg2://a:b@remote/db?fake=sslmode=verify-full"
    )
    with pytest.raises(ValueError):
        database_url()


def test_scheduler_enforces_concurrency_and_total_budget(env, monkeypatch):
    maximum, gets = 0, 0
    create, get = env.jobs.create, env.jobs.get

    async def counted_create(config, assignment):
        nonlocal maximum
        receipt = await create(config, assignment)
        active = sum(job.status.state == JobStatus.State.RUNNING for job in env.jobs.jobs.values())
        maximum = max(maximum, active)
        assert active <= config.concurrency
        return receipt

    async def finishing_get(job_id):
        nonlocal gets
        gets += 1
        if gets > 4:
            row = next(
                row for row in env.journal.rows(env.run["run_id"]) if row["job_id"] == job_id
            )
            if row["outcome"] is None:
                complete(env, row)
        return await get(job_id)

    async def no_wait(seconds):
        return None

    env.jobs.create, env.jobs.get = counted_create, finishing_get
    monkeypatch.setattr(asyncio, "sleep", no_wait)
    assert asyncio.run(env.coord.drive())
    assert maximum == env.config.concurrency
    assert env.jobs.creates == env.config.trials
    assert len(env.coord.study.trials) == env.config.trials


@pytest.mark.parametrize("event", ["after_ask", "before_create", "after_create"])
def test_process_kill_releases_lock_and_preserves_dispatch_identity(config, tmp_path, event):
    from datetime import datetime
    from datetime import timezone

    from conftest import FakeJobs
    from job_client import make_request
    import optuna

    url = os.environ.get("OPTUNA_TEST_DSN")
    if not url:
        pytest.skip("Set OPTUNA_TEST_DSN")
    marker = tmp_path / "accepted.json"
    source = """
import asyncio, json, os, signal
from dataclasses import asdict
from common import Config, digest
from coordinator import Coordinator
from journal import Journal
from pathlib import Path
import uuid
cfg=Config(**json.loads(os.environ["TEST_CONFIG"]))
j=Journal(os.environ["OPTUNA_TEST_DSN"])
j.initialize()
name="killed-"+uuid.uuid4().hex
class Jobs:
    async def create(self, config, assignment):
        Path(os.environ["TEST_MARKER"]).write_text(json.dumps(assignment))
        return {"job_id":"aijob-killed-fixture", "operation_id":"op-fixture"}
def checkpoint(event):
    if event == os.environ["TEST_EVENT"]:
        os.kill(os.getpid(), signal.SIGKILL)
with j.lock(name):
    run=j.create_run(name,asdict(cfg),digest(asdict(cfg)))
    print(json.dumps({"run_id":run["run_id"]}),flush=True)
    c=Coordinator(j,run,Jobs(),None,os.environ["OPTUNA_TEST_DSN"],checkpoint)
    asyncio.run(c.allocate())
"""
    child_env = {
        **os.environ,
        "TEST_CONFIG": json.dumps(asdict(config)),
        "TEST_MARKER": str(marker),
        "TEST_EVENT": event,
        "PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve().parents[1]),
    }
    process = subprocess.run(
        [sys.executable, "-c", source], env=child_env, capture_output=True, text=True, timeout=30
    )
    assert process.returncode == -9
    run_id = json.loads(process.stdout)["run_id"]
    journal = Journal(url)
    run = journal.run(run_id)
    jobs = FakeJobs()
    if marker.exists():
        assignment = json.loads(marker.read_text())
        request = make_request(config, assignment)
        request.metadata.id = "aijob-killed-fixture"
        request.metadata.created_at = datetime.now(timezone.utc)
        jobs.jobs[request.metadata.id] = Job(
            metadata=request.metadata,
            spec=request.spec,
            status=JobStatus(state=JobStatus.State.RUNNING),
        )
    try:
        with journal.lock(run["study_name"]):
            coord = Coordinator(journal, run, jobs, None, url)
            asyncio.run(coord.reconcile())
            row = journal.rows(run_id)[0]
            assert jobs.creates == 0
            if event == "after_ask":
                assert coord.study.trials[0].state == TrialState.FAIL
            elif event == "before_create":
                assert row["state"] == "DISPATCH_INTENT" and row["job_id"] is None
            else:
                assert row["job_id"] == "aijob-killed-fixture"
            optuna.delete_study(study_name=run["study_name"], storage=coord.storage)
            journal.query("DELETE FROM nebius_optuna.submissions WHERE run_id=%s", (run_id,))
            journal.query("DELETE FROM nebius_optuna.runs WHERE run_id=%s", (run_id,))
    finally:
        journal.close()


def test_worker_export_failure_never_completes_trial(env, monkeypatch):
    row = allocate(env)
    original = env.s3.put_object

    def fail_manifest(**kwargs):
        if kwargs["Key"].endswith("manifest.json"):
            raise OSError("simulated publication failure")
        return original(**kwargs)

    monkeypatch.setattr(env.s3, "put_object", fail_manifest)
    data = TensorDataset(torch.zeros(128, 1, 28, 28), torch.arange(128) % 10)
    with pytest.raises(OSError):
        run_worker(
            env.config,
            row["assignment"],
            env.journal,
            env.store,
            env.url,
            fixture=(data, data, {"fixture": True}),
        )
    assert env.coord.study.trials[0].state == TrialState.RUNNING
    with pytest.raises(Exception):
        env.store.verify(row["assignment"])
    env.jobs.jobs[row["job_id"]].status.state = JobStatus.State.FAILED
    assert asyncio.run(env.coord.reconcile())
    assert env.coord.study.trials[0].state == TrialState.FAIL


def test_worker_refuses_implicit_cpu_fallback(env, monkeypatch):
    row = allocate(env)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        run_worker(env.config, row["assignment"], env.journal, env.store, env.url)
    assert env.coord.study.trials[0].state == TrialState.RUNNING


def test_missing_job_record_is_not_proof_of_failure(env):
    row = allocate(env)
    del env.jobs.jobs[row["job_id"]]
    assert not asyncio.run(restart(env).reconcile())
    assert env.coord.study.trials[0].state == TrialState.RUNNING
    assert env.jobs.creates == 1


def test_cancelled_status_waits_for_known_operation(env):
    from types import SimpleNamespace

    row = allocate(env)
    asyncio.run(env.coord.cancel())
    original = env.jobs.operation

    async def pending(operation_id):
        return SimpleNamespace(finished_at=None, status=None)

    env.jobs.operation = pending
    assert not asyncio.run(env.coord.reconcile())
    assert env.coord.study.trials[0].state == TrialState.RUNNING
    env.jobs.operation = original
    assert asyncio.run(env.coord.reconcile())
    assert env.coord.study.trials[0].state == TrialState.FAIL
    assert env.journal.row(row["submission_id"])["observation"]["state"] == "CANCELLED"


def test_cli_failure_does_not_print_secret_values(config, tmp_path):
    import yaml

    path = tmp_path / "invalid.yaml"
    values = asdict(config)
    values["image"] = "sensitive-fixture-value"
    path.write_text(yaml.safe_dump(values))
    result = subprocess.run(
        [sys.executable, "nebius/coordinator.py", "preflight", "--config", str(path)],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "sensitive-fixture-value" not in result.stdout + result.stderr
    assert "ValueError" in result.stdout


def test_modified_duplicate_candidate_is_not_filtered_away(env):
    env.jobs.mode = "lost_response"
    row = allocate(env)
    original = next(iter(env.jobs.jobs.values()))
    duplicate = Job.from_json(original.to_json())
    duplicate.metadata.id = "aijob-modified-duplicate"
    duplicate.spec.args = "changed-after-create"
    env.jobs.jobs[duplicate.metadata.id] = duplicate
    assert not asyncio.run(restart(env).reconcile())
    assert env.journal.row(row["submission_id"])["job_id"] is None


def test_status_reads_journal_while_coordinator_holds_lock(env, tmp_path):
    import yaml

    row = allocate(env)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(asdict(env.config)))
    result = subprocess.run(
        [
            sys.executable,
            "nebius/coordinator.py",
            "status",
            "--config",
            str(path),
            "--run-id",
            env.run["run_id"],
        ],
        env={**os.environ, "OPTUNA_STORAGE_URL": env.url},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert row["job_id"] in result.stdout
    assert env.jobs.creates == 1


def test_unverified_external_completion_is_not_best(env):
    row = allocate(env)
    env.coord.study.tell(row["trial_number"], 0.99)
    assert env.coord.summary()["best_trial"] is None


def test_database_read_outage_does_not_become_artifact_failure(env, monkeypatch):
    row = allocate(env)
    complete(env, row)
    env.journal.query(
        "UPDATE nebius_optuna.submissions SET terminal_seen_at=now()-interval '1 day' "
        "WHERE submission_id=%s",
        (row["submission_id"],),
    )
    original = env.coord.trial

    def unavailable(number):
        raise OSError("database read unavailable")

    monkeypatch.setattr(env.coord, "trial", unavailable)
    assert not asyncio.run(env.coord.reconcile())
    assert env.journal.row(row["submission_id"])["outcome"] is None
    monkeypatch.setattr(env.coord, "trial", original)
    assert asyncio.run(env.coord.reconcile())
    assert env.coord.summary()["best_trial"] == row["trial_number"]


def test_resource_ids_survive_terminal_status(env):
    row = allocate(env)
    env.journal.update(row["submission_id"], observation={"instance_ids": ["compute-prior"]})
    complete(env, row)
    assert asyncio.run(env.coord.reconcile())
    assert env.journal.row(row["submission_id"])["observation"]["instance_ids"] == [
        "compute-prior"
    ]


def test_empty_minimizing_study_is_not_adopted(env):
    import optuna

    optuna.delete_study(study_name=env.run["study_name"], storage=env.coord.storage)
    study = optuna.create_study(
        study_name=env.run["study_name"], storage=env.coord.storage, direction="minimize"
    )
    with pytest.raises(ValueError, match="maximize"):
        restart(env)
    assert "nebius_example" not in study.user_attrs


def test_s3_read_outage_does_not_become_artifact_failure(env, monkeypatch):
    from botocore.exceptions import EndpointConnectionError

    row = allocate(env)
    complete(env, row)
    env.journal.query(
        "UPDATE nebius_optuna.submissions SET terminal_seen_at=now()-interval '1 day' "
        "WHERE submission_id=%s",
        (row["submission_id"],),
    )

    def unavailable(assignment):
        raise EndpointConnectionError(endpoint_url="https://s3.example.test")

    monkeypatch.setattr(env.store, "verify", unavailable)
    assert not asyncio.run(env.coord.reconcile())
    assert env.journal.row(row["submission_id"])["outcome"] is None
    assert env.coord.study.trials[0].state == TrialState.RUNNING


@pytest.mark.parametrize("length", [123, 127, 128, 129, 144])
def test_submission_image_byte_boundary(config, length):
    image = "r/" + "x" * (length - 74) + "@sha256:" + "a" * 64
    candidate = replace(config, image=image).validate()
    assert len(image.encode("utf-8")) == length
    if length <= 128:
        candidate.validate_submission()
    else:
        with pytest.raises(ImageReferenceTooLong, match="retain the full @sha256"):
            candidate.validate_submission()


def test_submission_image_limit_counts_bytes(config):
    image = "r/" + "é" * 29 + "@sha256:" + "a" * 64
    assert len(image) < 128 < len(image.encode("utf-8"))
    with pytest.raises(ImageReferenceTooLong):
        replace(config, image=image).validate().validate_submission()


@pytest.mark.parametrize("action", ["preflight", "run"])
def test_long_image_cli_fails_before_database_access(config, tmp_path, action):
    import yaml

    image = "private-sensitive-registry/" + "x" * 50 + "@sha256:" + "a" * 64
    path = tmp_path / "long.yaml"
    path.write_text(yaml.safe_dump(asdict(replace(config, image=image))))
    command = [sys.executable, "nebius/coordinator.py", action, "--config", str(path)]
    if action == "run":
        command += ["--study", "new-study", "--execute"]
    result = subprocess.run(
        command,
        env={k: v for k, v in os.environ.items() if k != "OPTUNA_STORAGE_URL"},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "ImageReferenceTooLong" in result.stdout
    assert "shorten the registry/repository path" in result.stdout
    assert "private-sensitive-registry" not in result.stdout + result.stderr


def test_long_image_allocation_has_no_side_effects(env):
    env.coord.config = replace(env.config, image="r/" + "x" * 60 + "@sha256:" + "a" * 64)
    with pytest.raises(ImageReferenceTooLong):
        allocate(env)
    assert env.journal.rows(env.run["run_id"]) == []
    assert env.coord.study.trials == []
    assert env.jobs.creates == 0


def test_legacy_long_image_can_reconcile_cancel_and_cleanup(env, monkeypatch):
    # Emulate a run submitted before the length guard, retaining its original config.
    legacy = replace(env.config, image="r/" + "x" * 60 + "@sha256:" + "a" * 64)
    env.run["config"] = asdict(legacy)
    env.run["config_hash"] = digest(asdict(legacy))
    owner = env.coord.study.user_attrs["nebius_example"]
    env.coord.study.set_user_attr(
        "nebius_example", {**owner, "config_hash": env.run["config_hash"]}
    )
    env.coord = restart(env)
    with monkeypatch.context() as patch:
        patch.setattr(Config, "validate_submission", lambda self: None)
        row = allocate(env)
    assert asyncio.run(restart(env).reconcile())
    assert not asyncio.run(env.coord.cancel())
    assert asyncio.run(env.coord.reconcile())
    assert env.journal.row(row["submission_id"])["outcome"]["state"] == "FAIL"
    assert not asyncio.run(env.coord.cleanup())
    assert asyncio.run(restart(env).cleanup())
    assert env.jobs.creates == env.jobs.cancels == env.jobs.deletes == 1
