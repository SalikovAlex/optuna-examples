# Distributed GPU tuning with Nebius Serverless Jobs

This example tunes a small FashionMNIST classifier with an external Optuna coordinator.
The coordinator allocates parameters with `Study.ask()`, submits one bounded Nebius
Serverless Job per trial, and uses `Study.tell()` after verifying the result. PostgreSQL
holds the shared study and a separate submission journal. Workers report intermediate
accuracy to the existing trial and can prune early.

This is an experimental example. Local PostgreSQL, simulated Jobs/S3, CPU training and
SDK transport tests are provided. Live validation on September 12, 2026 confirmed one
GPU trial using a **123-byte private digest-pinned image reference** through the production
coordinator: private image pull, CUDA training, secret delivery, PostgreSQL TLS, verified
model readback and independent VM/disk cleanup. Use a short repository name and retain
the full digest; the example rejects new submissions over 128 bytes. Earlier lifecycle
diagnostics used a tag override and have a narrower scope of evidence, described below.

## Architecture and scope

```text
external CPU coordinator --Create/Get/Cancel/Delete--> one GPU Job per trial
          |                                             |
          +----------- PostgreSQL study + journal ------+
          |                                             |
          +---- read and verify <---- S3 trial outputs --+
```

The example uses one coordinator, one dedicated study, trusted workers, a small explicit
concurrency limit, regular one-GPU Jobs and single-objective maximization. It does not
provision the coordinator host, PostgreSQL, a bucket, Compute VMs or Endpoints. There is
no Optuna core modification, new installable integration package, automatic retry,
checkpoint resume, multi-node training or independent cleanup watchdog.

The search space covers hidden width, dropout, learning rate and Adam/SGD. The objective
adapts [`pytorch_simple.py`](../pytorch/pytorch_simple.py), with all suggestions made by
the coordinator. The separate coordinator follows the
[Spark ask-and-tell example](../spark/ask_and_tell_spark.py); shared PostgreSQL also appears
in the [Kubernetes example](../kubernetes/README.md).

## Requirements

- Python 3.12 for the coordinator; install `pip install -r nebius/requirements.txt`.
- A durable PostgreSQL database reachable by both the coordinator and Jobs. Use a
  dedicated database for this example, with backups and sufficient connections. Each
  Optuna process can use up to three pooled connections; each journal uses one more.
- A Nebius project, subnet with outbound connectivity, available **one-GPU** platform/preset,
  sufficient Compute quotas, Object Storage bucket, and a registry for your worker image.
- A dedicated renewable Nebius service-account credential file for the coordinator.
  Workers do not receive credentials for creating/cancelling Jobs.
- Native secret **version IDs** for the worker database URL and S3 credentials; optional
  native registry credentials. No secret values belong in YAML, images or command arguments.

Coordinator environment (set the values securely outside shell history where appropriate):

| Variable | Purpose |
|---|---|
| `NEBIUS_CREDENTIALS_FILE` | Explicit Nebius SDK service-account credential file; no ambient CLI profile is selected |
| `OPTUNA_STORAGE_URL` | `postgresql+psycopg2://USER:PASSWORD@HOST/DB?sslmode=verify-full&sslrootcert=CA_PATH`; URL-encode special characters |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Coordinator's access to the configured output bucket |
| `AWS_SESSION_TOKEN` | Optional coordinator S3 session token |

Remote PostgreSQL URLs must use `sslmode=verify-full`. The CA path must exist on the
machine using that URL. Put the worker's URL in the database secret using a CA path
available **inside its image**; publicly trusted databases can use the image's CA bundle.
Private CAs require adding the public CA certificate to your image. A laptop-only
PostgreSQL service is not automatically reachable from a Nebius subnet.

The database secret needs the exact payload key `OPTUNA_STORAGE_URL`. The S3 secret
needs `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`; this worker configuration uses
static S3 access keys, not session-token secrets. Registry secrets use native
`registry_username`/`registry_password` keys, as documented by Serverless Jobs. The SDK references these secret versions;
it does not fetch payloads into the coordinator or put secrets in the injected assignment.
Verify the service account's Jobs, subnet, registry and secret-access permissions before
a live run. Public Nebius guides document an `editor` prerequisite; a minimal IAM policy
has not been live-tested here.

Optuna's RDB interface is not a per-trial security boundary. Workers need access to
study history, intermediate values and the example journal. Do not share this database
with untrusted training code or unrelated tenants.

## Build the worker and configure the run

From the repository root:

```bash
docker build --platform linux/amd64 -t YOUR_REGISTRY/optuna:YOUR_TAG nebius
```

The Dockerfile pins a Linux amd64 Python base and installs Optuna 5.0.0, PyTorch 2.14.0 and
torchvision 0.29.0. Linux PyPI PyTorch wheels include CUDA userspace libraries (CUDA 13.0
in the locally built image); the cloud runtime must supply a compatible NVIDIA driver.
Validate that driver/GPU combination before a paid sweep. The worker refuses implicit
CPU fallback. Transitive dependencies are constrained for the worker image.

The build downloads FashionMNIST from a pinned upstream commit, checks its published
MD5 checksums using torchvision, and records SHA-256 hashes of the compressed and
uncompressed files. Workers validate those files before use; no dataset download is
needed during a Job. FashionMNIST is [MIT licensed](https://github.com/zalandoresearch/fashion-mnist/blob/b2617bb6d3ffa2e429640350f613e3291e10b141/LICENSE).
Retain that attribution if adapting the image. There is no pretrained model or model API.

Push your image only when authorized, resolve its **registry manifest digest**, and put
`registry/path@sha256:...` in a private copy of `nebius/config.example.yaml`. A local
Docker image ID is not a registry manifest digest. Never use a mutable tag in the run
configuration. Keep the complete reference at most **128 UTF-8 bytes**: Serverless copies
it into a Compute label. Use a short repository such as `optuna`; even `optuna-worker`
can exceed the limit with a private registry hostname and full digest. The example
rejects longer references before allocating a trial; it never truncates the digest or
falls back to a tag. Existing runs remain readable for reconciliation and cleanup;
changing their image requires a new study.

Fill every `REPLACE_...` value, including the exact one-GPU preset.

```bash
python nebius/coordinator.py preflight --config /path/to/my-config.yaml
```

Preflight validates configuration and initializes the example's PostgreSQL schema. It
does not contact Nebius, reserve quota, inspect an image, confirm networking or prove
GPU availability. Review the total Job count, concurrency, disk size and deadlines
before executing a cloud command.

## Start, observe and recover

These commands contact Nebius only with explicit `--execute`:

```bash
python nebius/coordinator.py run --config /path/to/my-config.yaml --study my-new-study --execute
```

Save the printed run ID. The YAML fixes the total allocation budget (default eight),
concurrency (default two), epoch/sample bounds, one-hour provider timeout, 15-minute
worker deadline and two-hour run deadline. **Failed/pruned/interrupted allocations
consume the trial budget**; the coordinator does not keep launching until it gets eight
successful trials. Provider provisioning time and cancellation latency mean these settings
are not an exact spending cap. Serverless consumes Compute quotas and storage remains
billable after Jobs finish. Agree a separate financial cap before a live run.

```bash
python nebius/coordinator.py status --config /path/to/my-config.yaml --run-id RUN_ID
python nebius/coordinator.py reconcile --config /path/to/my-config.yaml --run-id RUN_ID --execute
python nebius/coordinator.py resume --config /path/to/my-config.yaml --run-id RUN_ID --execute
```

`status` reads the durable journal without cloud credentials and can run while the
coordinator holds its lock. It reports the last reconciled state, not a fresh provider read. `reconcile` never creates
Jobs or allocates trials. It imports verified outcomes and attempts to attach an existing
Job after a lost response. `resume` is the explicit request to continue remaining
allocations after reconciliation. The configuration and study ownership must match the
original run; use a new study for a changed configuration. Keep other Optuna writers,
`enqueue_trial()` and retry callbacks away from this dedicated study.

Exit code 2 means an error, pending cancellation/deletion, a stopped/expired run or an
unresolved observation. Inspect the printed per-trial state and operation/Job IDs. A
successful reconciliation command can still report active Jobs: it means the observations
were reconciled, not that every trial completed. If the coordinator exits or is killed,
Jobs can continue independently and finish within their own bounds. Re-run reconciliation;
do not start another sweep as a recovery mechanism.

For logs, use the Nebius console or the documented CLI:

```bash
nebius ai job logs JOB_ID --follow --timestamps
```

Logs are diagnostic, not the objective-value transport. The worker emits run/trial IDs
and epoch accuracy to stdout. A log disconnection does not change the Optuna outcome.
SDK/SQL exceptions are printed as error types rather than raw text because raw exceptions
can contain credentials. Provider state/code and exposed instance IDs are retained in
the journal. Optuna Dashboard can use the same study database if independently configured.

## Trial outcomes and artifacts

Each submission writes under
`s3://BUCKET/PREFIX/RUN_ID/trials/TRIAL_NUMBER/SUBMISSION_ID/`. Outputs include model weights
for COMPLETE trials, metrics, dataset provenance, `result.json`, and a manifest written
last. The coordinator downloads the listed bytes, checks lengths/SHA-256, identity,
runtime provenance and intermediate metrics before telling Optuna a result. Incomplete
or corrupt uploads remain pending briefly, then fail the trial. No shared `best/model.pt`
path is overwritten. The summary includes each manifest URI and the best completed
trial number recorded as finalized in the journal. Database backup and artifact retention are the user's responsibility.

| Provider/application observation | Optuna result |
|---|---|
| `COMPLETED` with valid COMPLETE output | COMPLETE with finite validation accuracy |
| `COMPLETED` with verified pruning decision/output | PRUNED; a pruned worker exits normally |
| `FAILED`, `ERROR`, confirmed `CANCELLED` | FAIL with diagnostic code; never a fabricated low score |
| `COMPLETED` with invalid output after the verification window | FAIL |
| Uncertain Create, missing Job record, read outage or unknown status | Remains unresolved/RUNNING; no replacement Job |

The worker resolves the existing study-local trial number through documented
`RDBStorage` methods, reports intermediate values, and calls the configured MedianPruner
on a fresh FrozenTrial. It never allocates another trial or constructs a private trial
ID. Optuna's automatic storage heartbeat cleanup does **not** work with ask-and-tell;
this example uses its journal/provider reconciliation instead. Pruning is statistical,
not a user cancellation or infrastructure failure. Asynchronous completion and sampler
restart can change subsequent suggestions despite a fixed seed.

## Ambiguous Create, cancellation and cleanup

The coordinator commits `DISPATCH_INTENT` **before** Create and uses `retries=0` for SDK
mutations. Its persisted submission UUID is also the idempotency header. This does not
assume an undocumented server deduplication lifetime: a lost response is never replayed.
Reconciliation exhausts Job pagination and compares labels, project, name, creation time
and visible configuration. Job names alone are not unique identities. Zero candidates,
multiple matching candidates or insufficient information require operator investigation;
zero matches are not proof that creation failed. All such submissions hold their slots
and halt new submissions.

Only one coordinator may hold the study's PostgreSQL advisory lock. It uses the same
non-reconnecting connection for journal writes. A lost connection stops dispatch; the
submission intent protects the crash boundary. This is not an HA scheduler or an
exactly-once execution guarantee. Interrupted allocations before dispatch may fail without
having created any Job. A worker's durable one-time claim prevents duplicate training
processes from evaluating the same assignment.

```bash
python nebius/coordinator.py cancel --config /path/to/my-config.yaml --run-id RUN_ID --execute
python nebius/coordinator.py reconcile --config /path/to/my-config.yaml --run-id RUN_ID --execute
python nebius/coordinator.py cleanup --config /path/to/my-config.yaml --run-id RUN_ID --execute
```

Stop the running coordinator (Ctrl-C) before a mutating recovery command so it releases
the study lock; existing Jobs continue until cancelled or terminal.

Cancel permanently stops further allocation for that run. It sends Cancel only for
verified owned Job IDs; cancellation acceptance is not terminal confirmation. Reconcile
until the provider is terminal and any known cancellation operation has completed.
Neither cancellation nor deletion with a lost acknowledgment is automatically retried.
A completion/cancellation race retains a valid completed result if the provider completed.

Cleanup deletes only finalized, terminal Job **records**, then requires a successful
Delete operation and NotFound on a subsequent invocation. Repeat cleanup to observe
operation completion, not to replay its mutation. It never deletes study data, buckets,
filesystems, registry images or model outputs. The database keeps Job/operation/instance
IDs even after metadata deletion. An unexplained NotFound before a recorded Delete is
an unresolved observation.

**After coordinator loss there is no independent watchdog.** Provider/worker deadlines
reduce runaway execution but do not guarantee immediate provisioning/cancellation or
complete VM/disk cleanup. Inspect retained instance IDs and obtain resource deletion
confirmation through authorized platform checks/support; a deleted Job record does not
prove every underlying resource disappeared. Do not remove diagnostic records before
accounting for these resources. Cancellation can destroy local state without a final
checkpoint; only already published outputs are retained.

## Local validation

Use a disposable PostgreSQL database. The suite creates/deletes its own studies and
journal rows and uses Moto S3 and local gRPC servers; it does not need cloud credentials.

```bash
pip install -r nebius/requirements-test.txt
# Set OPTUNA_TEST_DSN to the disposable postgresql+psycopg2 URL.
python -m pytest nebius/tests -q
black nebius --check
flake8 nebius
isort nebius --check
```

Without `OPTUNA_TEST_DSN`, database tests are skipped; that is not a complete validation.
The GitHub Actions workflow sets it explicitly. CPU fixtures exercise actual PyTorch
training, model loading, Optuna reporting/pruning and S3 artifact verification. Production
verification rejects CPU fixture manifests. Tests also kill coordinator subprocesses
around dispatch boundaries and verify real SDK retry/pagination behavior against gRPC
stubs. None establishes real cloud behavior.

## Live validation notes (September 12, 2026)

An initial private registry image reference of 144 bytes was rejected with SDK 0.6.10
and `INVALID_ARGUMENT`: Compute reported a label value length of 144 exceeding 64.
A control with no custom labels produced the same error. A short tag pointing to the
identical manifest was accepted. Subsequent source inspection identified a **128-byte**
label value limit; the error incorrectly reports the 64-byte key limit. Serverless copies
the full image reference into this label.

The integration workaround uses a short `optuna` repository while retaining the full
SHA-256 digest. A follow-up run submitted a 123-byte private reference without changing
the production request. The provider's Job record matched that exact reference. One L40S
trial completed five CUDA epochs; downloaded weights reproduced **992/1280 = 0.775** on
an independent CPU evaluation, with matching dataset hashes. Five S3 objects passed
length/SHA-256 checks, and the recorded VM and boot disk independently returned NotFound.
The temporary coordinator connection used a DNS-verified address to bypass a local
negative DNS cache, retaining `sslmode=verify-full` and hostname verification. The worker
used its normal private database hostname. No Nebius backend change was needed for this
short-reference path; longer references remain subject to the provider issue.

The diagnostic harness replaced only the outgoing image reference with that tag, while
leaving the production example's digest requirement unchanged. Registry manifest checks
support identity at those observation times; they do not provide immutable binding during
execution. The GPU worker used PyTorch 2.14.0+cu130 on L40S. An independent model download,
checksum check and CPU evaluation reproduced 992/1280 validation accuracy (0.775). A
four-trial sweep returned three COMPLETE trials and one PRUNED trial. The live exercise
also confirmed worker timeout/nonzero exit, denied S3 upload and recovery after SIGKILL
between successful Create and receipt persistence. A separate CPU control reached its
one-hour provider timeout. Cancellation while training reached CANCELLED and Optuna FAIL,
with its VM and disk independently confirmed absent. The exercise exposed SDK operation
property/method mismatches, now covered by transport tests. A short-job boundary probe
observed RUNNING, then found COMPLETED during cancellation and retained the verified
result without a Cancel RPC. It did not exercise an in-flight Cancel losing to completion.

The follow-up establishes the digest-pinned happy path, including image pull, training,
model verification and VM/disk cleanup. It does not repeat the earlier pruning, failure,
timeout and recovery matrix with digest-pinned requests. An in-flight Cancel losing to
completion and the optional registry-auth secret path remain untested live. Keep this
scope distinction when assessing broader acceptance; retain exact image and SDK/framework
versions and redacted resource evidence. Stop on uncertain Create or unresolved cleanup
instead of increasing the number of probes.

References: [Jobs](https://docs.nebius.com/serverless/jobs/manage),
[lifecycle](https://docs.nebius.com/serverless/lifecycle),
[quotas](https://docs.nebius.com/serverless/pricing-quotas),
[Optuna storage](https://optuna.readthedocs.io/en/v5.0.0/reference/generated/optuna.storages.RDBStorage.html).
