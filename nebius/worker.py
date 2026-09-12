"""Evaluate one preassigned trial. The command-line worker requires CUDA."""

from dataclasses import asdict
import io
import json
import logging
import math
from pathlib import Path
import sys
import time

from common import Artifacts
from common import Config
from common import database_url
from common import digest
from common import encode
from common import identity
from common import PROTOCOL
from common import pruner
from common import sha256
from common import study_storage
from journal import Journal
import optuna
from optuna.trial import TrialState
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms


def model_for(params):
    # Adapted from pytorch/pytorch_simple.py, with all suggestions made by the coordinator.
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, params["width"]),
        nn.ReLU(),
        nn.Dropout(params["dropout"]),
        nn.Linear(params["width"], 10),
    )


def load_data(root, config):
    raw = Path(root, "data.json").read_bytes()
    manifest = json.loads(raw)
    from prepare_data import REVISION

    if manifest["source_revision"] != REVISION or len(manifest["files"]) != 8:
        raise ValueError("Dataset revision/file count mismatch")
    for name, expected in manifest["files"].items():
        path = Path(root, name).resolve()
        if not path.is_relative_to(Path(root).resolve()) or sha256(path.read_bytes()) != expected:
            raise ValueError("Dataset checksum mismatch")
    train = datasets.FashionMNIST(
        root, train=True, download=False, transform=transforms.ToTensor()
    )
    valid = datasets.FashionMNIST(
        root, train=False, download=False, transform=transforms.ToTensor()
    )
    return (
        Subset(train, range(config.train_samples)),
        Subset(valid, range(config.valid_samples)),
        manifest,
    )


def report(storage, study, trial_id, step, accuracy):
    if type(step) is not int or step < 0 or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Invalid epoch metric")
    frozen = storage.get_trial(trial_id)
    if frozen.state != TrialState.RUNNING:
        raise ValueError("Assigned trial is no longer running")
    values = frozen.intermediate_values
    if step in values:
        if values[step] != accuracy:
            raise ValueError("Conflicting duplicate epoch metric")
    else:
        if step != len(values):
            raise ValueError("Out-of-order epoch metric")
        storage.set_trial_intermediate_value(trial_id, step, accuracy)
    return study.pruner.prune(study, storage.get_trial(trial_id))


def run_worker(config, assignment, journal, store, url, data_root="/data", fixture=None):
    started = time.monotonic()
    if assignment.get("protocol") != PROTOCOL or assignment["params_hash"] != digest(
        assignment["params"]
    ):
        raise ValueError("Invalid assignment")
    run = journal.run(assignment["run_id"])
    row = journal.row(assignment["submission_id"])
    if (
        row["assignment"] != assignment
        or run["config_hash"] != digest(asdict(config))
        or run["stopped"]
        or run["expired"]
    ):
        raise ValueError("Assignment does not match an active run")
    storage = study_storage(url)
    study = optuna.load_study(
        study_name=assignment["study_name"], storage=storage, pruner=pruner()
    )
    if study.user_attrs.get("nebius_example") != {
        "protocol": PROTOCOL,
        "run_id": run["run_id"],
        "config_hash": run["config_hash"],
    }:
        raise ValueError("Study ownership mismatch")
    trial_id = storage.get_trial_id_from_study_id_trial_number(
        storage.get_study_id_from_name(study.study_name), assignment["trial_number"]
    )
    trial = storage.get_trial(trial_id)
    if trial.state != TrialState.RUNNING or digest(trial.params) != assignment["params_hash"]:
        raise ValueError("Trial identity/parameters mismatch")
    if not journal.claim(assignment["submission_id"]):
        raise ValueError("Submission already claimed, cancelled, or finalized")
    if fixture is None and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no implicit CPU fallback")
    device = torch.device("cuda" if fixture is None else "cpu")
    train, valid, data_manifest = load_data(data_root, config) if fixture is None else fixture
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    model = model_for(assignment["params"]).to(device)
    optimizer = getattr(torch.optim, assignment["params"]["optimizer"])(
        model.parameters(), lr=assignment["params"]["lr"]
    )
    train_loader = DataLoader(
        train,
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=0,
    )
    valid_loader = DataLoader(valid, batch_size=128, shuffle=False, num_workers=0)
    metrics, state = [], "COMPLETE"
    for epoch in range(config.epochs):
        if journal.row(assignment["submission_id"])["cancel_intent"]:
            raise RuntimeError("Cancellation requested")
        model.train()
        for data, target in train_loader:
            if time.monotonic() - started >= config.worker_timeout:
                raise TimeoutError("Worker wall deadline")
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(data.to(device)), target.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        correct, count = 0, 0
        with torch.no_grad():
            for data, target in valid_loader:
                if time.monotonic() - started >= config.worker_timeout:
                    raise TimeoutError("Worker wall deadline")
                logits = model(data.to(device))
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite validation output")
                correct += (logits.argmax(dim=1).cpu() == target).sum().item()
                count += len(target)
        accuracy = correct / count
        metrics.append({"epoch": epoch, "accuracy": accuracy})
        print(json.dumps({**identity(assignment), **metrics[-1]}), flush=True)
        if report(storage, study, trial_id, epoch, accuracy):
            state = "PRUNED"
            journal.update(assignment["submission_id"], prune_intent=True)
            break
    result = {
        "identity": identity(assignment),
        "state": state,
        "value": accuracy,
        "epochs": len(metrics),
        "device": "cuda" if fixture is None else "cpu-fixture",
        "torch_version": torch.__version__,
        "optuna_version": optuna.__version__,
        "image": config.image,
        "data_hash": digest(data_manifest),
    }
    files = {"metrics.json": encode(metrics), "data.json": encode(data_manifest)}
    if state == "COMPLETE":
        buffer = io.BytesIO()
        torch.save({key: value.cpu() for key, value in model.state_dict().items()}, buffer)
        files["model.pt"] = buffer.getvalue()
    store.publish(assignment, result, files)
    return result


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("nebius").setLevel(logging.CRITICAL)
    journal = None
    try:
        payload = json.loads(Path(sys.argv[1]).read_bytes())
        config = Config(**payload["config"]).validate()
        url = database_url()
        journal = Journal(url)
        run_worker(config, payload["assignment"], journal, Artifacts(config), url)
    except Exception as error:
        # No raw SDK/SQL errors or environment values in logs.
        print(json.dumps({"error_type": type(error).__name__}), flush=True)
        raise SystemExit(1)
    finally:
        if journal:
            journal.close()
