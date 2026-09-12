"""External coordinator: one bounded Serverless Job per assigned Optuna trial."""

import argparse
import asyncio
from dataclasses import asdict
import json
import logging

from botocore.exceptions import BotoCoreError
from botocore.exceptions import ClientError
from common import Artifacts
from common import Config
from common import database_url
from common import digest
from common import ImageReferenceTooLong
from common import prefix_for
from common import PROTOCOL
from common import pruner
from common import study_storage
from common import TERMINAL
from job_client import Jobs
from job_client import matches
from job_client import Missing
from job_client import Rejected
from journal import Journal
import optuna
from optuna.trial import TrialState


class Coordinator:
    def __init__(self, journal, run, jobs, artifacts, url, checkpoint=None):
        self.journal, self.run, self.jobs, self.artifacts = journal, run, jobs, artifacts
        self.config = Config(**run["config"]).validate()
        if digest(run["config"]) != run["config_hash"]:
            raise ValueError("Stored config hash mismatch")
        self.checkpoint = checkpoint or (lambda event: None)
        self.storage = study_storage(url)
        self.study = optuna.create_study(
            study_name=run["study_name"],
            storage=self.storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.seed),
            pruner=pruner(),
        )
        owner = self.study.user_attrs.get("nebius_example")
        expected = {
            "protocol": PROTOCOL,
            "run_id": run["run_id"],
            "config_hash": run["config_hash"],
        }
        if self.study.direction.name != "MAXIMIZE":
            raise ValueError("Study must maximize the objective")
        if owner is None and not self.study.trials:
            self.study.set_user_attr("nebius_example", expected)
        elif owner != expected:
            raise ValueError("Study is not owned by this example/run")

    def row_update(self, row, **fields):
        self.journal.update(row["submission_id"], **fields)

    def trial(self, number):
        sid = self.storage.get_study_id_from_name(self.study.study_name)
        tid = self.storage.get_trial_id_from_study_id_trial_number(sid, number)
        return self.storage.get_trial(tid)

    def finish(self, row, state, value=None, manifest_hash=None, reason=None):
        outcome = {
            "state": state,
            "value": value,
            "manifest_hash": manifest_hash,
            "reason": reason,
        }
        if row["outcome"] is not None and row["outcome"] != outcome:
            raise ValueError("Conflicting outcome; manual investigation required")
        # Commit result evidence before tell: a crash after tell can safely replay this intent.
        self.row_update(row, outcome=outcome)
        self.checkpoint("before_tell")
        trial = self.trial(row["trial_number"])
        if trial.state.is_finished():
            expected_value = value if state == "COMPLETE" else trial.value
            if trial.state.name != state or trial.value != expected_value:
                raise ValueError("Optuna result conflicts with journal")
        else:
            self.study.tell(
                row["trial_number"],
                values=value if state == "COMPLETE" else None,
                state=TrialState[state],
            )
        self.checkpoint("after_tell")
        self.row_update(row, state="FINAL", diagnostic=reason)

    def recover_allocations(self):
        rows = self.journal.rows(self.run["run_id"])
        assigned = {r["trial_number"] for r in rows if r["trial_number"] is not None}
        allocation_numbers = {r["expected_number"] for r in rows if r["state"] == "ALLOCATING"}
        if any(t.number not in assigned | allocation_numbers for t in self.study.trials):
            raise ValueError("Unjournaled trial: another writer may be using this study")
        for row in rows:
            if row["state"] == "ALLOCATING":
                try:
                    self.trial(row["expected_number"])
                except KeyError:
                    self.row_update(row, state="ABANDONED", diagnostic="allocation_not_created")
                    continue
                self.row_update(row, trial_number=row["expected_number"])
                row["trial_number"] = row["expected_number"]
                self.finish(row, "FAIL", reason="allocation_interrupted")
            elif row["state"] == "PREPARED":
                # A recovered PREPARED row never reached dispatch; do not silently launch it.
                self.finish(row, "FAIL", reason="prepared_not_dispatched")

    async def allocate(self):
        self.config.validate_submission()
        row = self.journal.allocate(self.run["run_id"], len(self.study.trials), self.config.trials)
        self.checkpoint("before_ask")
        trial = self.study.ask()
        self.checkpoint("after_ask")
        if trial.number != row["expected_number"]:
            raise ValueError("Another writer allocated an unexpected trial")
        params = {
            "width": trial.suggest_int("width", 32, 128, step=32),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
        }
        assignment = {
            "protocol": PROTOCOL,
            "run_id": self.run["run_id"],
            "submission_id": row["submission_id"],
            "trial_number": trial.number,
            "study_name": self.study.study_name,
            "params": params,
            "params_hash": digest(params),
        }
        self.row_update(row, trial_number=trial.number, assignment=assignment, state="PREPARED")
        self.checkpoint("prepared")
        if not self.journal.dispatch(row["submission_id"]):
            raise RuntimeError("Submission was already dispatched")
        self.checkpoint("before_create")
        self.journal.query("SELECT 1")  # Fail closed if the lock-holding connection was lost.
        try:
            receipt = await self.jobs.create(self.config, assignment)
        except Rejected as exc:
            row = self.journal.row(row["submission_id"])
            self.finish(row, "FAIL", reason="admission_" + str(exc))
            return
        except Exception as exc:
            self.row_update(row, state="SUBMISSION_UNKNOWN", diagnostic=type(exc).__name__)
            return
        self.checkpoint("after_create")
        self.row_update(row, state="SUBMITTED", **receipt)
        self.checkpoint("receipt_saved")

    async def locate(self, row):
        if row["job_id"]:
            job = await self.jobs.get(row["job_id"])
        else:
            if row["operation_id"]:
                operation = await self.jobs.operation(row["operation_id"])
                if operation.resource_id:
                    job = await self.jobs.get(operation.resource_id)
                else:
                    job = None
            else:
                job = None
            if job is None:
                candidates = await self.jobs.candidates(self.config, row)
                if len(candidates) != 1:
                    raise ValueError(
                        "Submission unresolved: candidate count=" + str(len(candidates))
                    )
                job = candidates[0]
            if not matches(job, self.config, row):
                raise ValueError("Job ownership/spec mismatch")
            self.row_update(row, job_id=job.metadata.id, state="SUBMITTED")
            row["job_id"] = job.metadata.id
        if not matches(job, self.config, row):
            raise ValueError("Job ownership/spec mismatch")
        self.row_update(
            row,
            observation={
                "state": job.status.state.name,
                "code": job.status.state_details.code,
                "instance_ids": sorted(
                    set((row["observation"] or {}).get("instance_ids", []))
                    | {i.compute_instance_id for i in job.status.instances}
                ),
            },
        )
        return job

    async def reconcile(self):
        self.recover_allocations()
        unresolved = False
        for row in self.journal.rows(self.run["run_id"]):
            if row["state"] in {"FINAL", "ABANDONED"}:
                continue
            if row["outcome"] is not None:
                self.finish(row, **row["outcome"])
                continue
            try:
                job = await self.locate(row)
                state = job.status.state.name
                if state not in TERMINAL:
                    if state not in {
                        "PROVISIONING",
                        "STARTING",
                        "IMAGE_PULLING",
                        "RUNNING",
                        "CANCELLING",
                    }:
                        raise ValueError("Unknown or deleting Job state")
                    continue
                if state == "CANCELLED" and row["cancel_operation"]:
                    operation = await self.jobs.operation(row["cancel_operation"])
                    if (
                        operation.finished_at is None
                        or operation.status is None
                        or operation.status.code.name != "OK"
                    ):
                        raise ValueError("Cancellation operation not confirmed successful")
                if state != "COMPLETED":
                    self.finish(row, "FAIL", reason=state + ":" + job.status.state_details.code)
                    continue
                age = self.journal.terminal_age(row["submission_id"])
                # Database read failures must remain unresolved, even after artifact grace.
                trial = self.trial(row["trial_number"])
                prune_intent = self.journal.row(row["submission_id"])["prune_intent"]
                try:
                    result, manifest_hash = self.artifacts.verify(row["assignment"])
                    if (
                        len(trial.intermediate_values) != result["epochs"]
                        or trial.intermediate_values[result["epochs"] - 1] != result["value"]
                    ):
                        raise ValueError("Stored intermediate metrics disagree with output")
                    if result["state"] == "PRUNED" and not prune_intent:
                        raise ValueError("No recorded pruning decision")
                except (BotoCoreError, ClientError) as exc:
                    # Missing output has a grace window; access/network outages do not
                    # establish that the worker produced an invalid result.
                    if not isinstance(exc, ClientError) or exc.response["Error"]["Code"] not in {
                        "NoSuchKey",
                        "404",
                        "NotFound",
                    }:
                        raise
                    if age < self.config.artifact_grace:
                        raise
                    self.finish(row, "FAIL", reason="artifact_verification_failed")
                    continue
                except Exception:
                    if age < self.config.artifact_grace:
                        raise
                    self.finish(row, "FAIL", reason="artifact_verification_failed")
                    continue
                self.finish(
                    row,
                    result["state"],
                    value=result["value"] if result["state"] == "COMPLETE" else None,
                    manifest_hash=manifest_hash,
                )
            except Exception as exc:
                self.row_update(row, diagnostic=type(exc).__name__)
                unresolved = True
        return not unresolved

    async def cancel(self):
        self.journal.stop(self.run["run_id"])
        ok = True
        for row in self.journal.rows(self.run["run_id"]):
            if row["state"] in {"FINAL", "ABANDONED", "ALLOCATING", "PREPARED"}:
                continue
            try:
                job = await self.locate(row)
                if job.status.state.name in TERMINAL:
                    continue
                if not row["cancel_intent"]:
                    self.row_update(row, cancel_intent=True)
                    operation = await self.jobs.cancel(job.metadata.id)
                    self.row_update(row, cancel_operation=operation)
                # Never repeat a cancellation whose response was lost.
                ok = False
            except Exception as exc:
                self.row_update(row, diagnostic=type(exc).__name__)
                ok = False
        return ok

    async def cleanup(self):
        ok = True
        for row in self.journal.rows(self.run["run_id"]):
            if row["state"] == "ABANDONED" or (row["state"] == "FINAL" and not row["job_id"]):
                continue
            if row["deleted"]:
                continue
            if row["state"] != "FINAL":
                ok = False
                continue
            try:
                if row["delete_intent"]:
                    if not row["delete_operation"]:
                        raise ValueError(
                            "Unknown Delete response; inspect before further mutation"
                        )
                    op = await self.jobs.operation(row["delete_operation"])
                    if op.finished_at is None or op.status is None or op.status.code.name != "OK":
                        raise ValueError("Delete operation not confirmed successful")
                    try:
                        await self.jobs.get(row["job_id"])
                    except Missing:
                        self.row_update(row, deleted=True, diagnostic=None)
                        continue
                    raise ValueError("Job record still present")
                job = await self.locate(row)
                if job.status.state.name not in TERMINAL:
                    raise ValueError("Refusing to delete a nonterminal Job")
                self.row_update(row, delete_intent=True)
                operation = await self.jobs.delete(job.metadata.id)
                self.row_update(row, delete_operation=operation)
                ok = False
            except Exception as exc:
                self.row_update(row, diagnostic=type(exc).__name__)
                ok = False
        return ok

    async def drive(self):
        while True:
            healthy = await self.reconcile()
            run = self.journal.run(self.run["run_id"])
            if run["expired"] or run["stopped"]:
                await self.cancel()
                await self.reconcile()
                return False
            rows = self.journal.rows(run["run_id"])
            active = [r for r in rows if r["state"] not in {"FINAL", "ABANDONED"}]
            if not healthy or any(
                r["state"] in {"DISPATCH_INTENT", "SUBMISSION_UNKNOWN"} for r in active
            ):
                return False
            if len(rows) >= self.config.trials and not active:
                return True
            if len(rows) < self.config.trials and len(active) < self.config.concurrency:
                await self.allocate()
            else:
                await asyncio.sleep(self.config.poll_seconds)

    def summary(self):
        return run_summary(self.journal, self.run, self.config)


def run_summary(journal, run, config):
    rows = journal.rows(run["run_id"])
    trials = [
        {
            "trial_number": r["trial_number"],
            "submission_id": r["submission_id"],
            "state": r["state"],
            "job_id": r["job_id"],
            "operation_id": r["operation_id"],
            "outcome": r["outcome"],
            "diagnostic": r["diagnostic"],
            "cancel_operation": r["cancel_operation"],
            "delete_operation": r["delete_operation"],
            "job_record_deleted": r["deleted"],
            "provider_observation": r["observation"],
            "manifest_uri": (
                (
                    "s3://"
                    + config.bucket
                    + "/"
                    + prefix_for(config, r["assignment"])
                    + "manifest.json"
                )
                if r["assignment"]
                else None
            ),
        }
        for r in rows
    ]
    complete = [r for r in rows if r["state"] == "FINAL" and r["outcome"]["state"] == "COMPLETE"]
    return {
        "run_id": run["run_id"],
        "study": run["study_name"],
        "trials": trials,
        "best_trial": (
            max(complete, key=lambda r: r["outcome"]["value"])["trial_number"]
            if complete
            else None
        ),
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["preflight", "run", "resume", "reconcile", "cancel", "cleanup", "status"],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--study")
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true", help="Allow cloud API access")
    args = parser.parse_args()
    config = Config.read(args.config)
    if args.action not in {"preflight", "status"} and not args.execute:
        parser.error("Cloud actions require --execute; review the config and limits first")
    if args.action == "run" and (not args.study or args.run_id):
        parser.error("run requires a new --study, without --run-id")
    if args.action not in {"preflight", "run"} and not args.run_id:
        parser.error("This action requires --run-id")
    if args.action in {"preflight", "run"}:
        config.validate_submission()
    journal = Journal(database_url())
    jobs = None
    try:
        if args.action != "status":
            journal.initialize()
        if args.action == "preflight":
            print("Config and PostgreSQL connection valid; cloud/image/GPU access not checked")
            return 0
        if args.action == "run":
            with journal.lock(args.study):
                run = journal.create_run(args.study, asdict(config), digest(asdict(config)))
        else:
            run = journal.run(args.run_id)
        if run["config_hash"] != digest(asdict(config)):
            raise ValueError("Config differs from the immutable run configuration")
        print(json.dumps({"run_id": run["run_id"], "study": run["study_name"]}), flush=True)
        if args.action == "status":
            print(json.dumps(run_summary(journal, run, config), indent=2))
            return 0
        with journal.lock(run["study_name"]):
            jobs = Jobs()
            coordinator = Coordinator(
                journal, run, jobs, Artifacts(config) if jobs else None, database_url()
            )
            if args.action in {"run", "resume"}:
                ok = await coordinator.drive()
            elif args.action == "reconcile":
                ok = await coordinator.reconcile()
            elif args.action == "cancel":
                ok = await coordinator.cancel()
                await coordinator.reconcile()
            elif args.action == "cleanup":
                ok = await coordinator.cleanup()
            else:
                ok = True
            print(json.dumps(coordinator.summary(), indent=2))
            return 0 if ok else 2
    finally:
        if jobs:
            await jobs.close()
        journal.close()


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("nebius").setLevel(logging.CRITICAL)
    try:
        raise SystemExit(asyncio.run(main()))
    except (Exception, KeyboardInterrupt) as error:
        # DSNs and SDK exceptions may contain credentials. IDs were printed before dispatch.
        print(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "action": (
                        str(error)
                        if isinstance(error, ImageReferenceTooLong)
                        else "Stop; inspect status/reconcile using the saved run ID"
                    ),
                }
            )
        )
        raise SystemExit(2)
