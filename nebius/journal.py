"""Example-owned PostgreSQL journal, separate from Optuna's tables.

A coordinator uses one non-reconnecting connection for both its advisory lock
and journal writes. Losing that connection stops mutation. Workers use their own
connection and claim exactly one submission before doing GPU work.
"""

from contextlib import contextmanager
import hashlib
import uuid

import psycopg2
from psycopg2.extras import Json
from psycopg2.extras import RealDictCursor
from sqlalchemy.engine import make_url


class Journal:
    def __init__(self, url):
        parsed = make_url(url)
        args = parsed.translate_connect_args(database="dbname", username="user")
        args.update(parsed.query)
        self.connection = psycopg2.connect(
            **args, connect_timeout=10, options="-c statement_timeout=15000"
        )
        self.connection.autocommit = True

    def close(self):
        self.connection.close()

    def query(self, sql, args=()):
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql, args)
            return [dict(row) for row in cursor.fetchall()] if cursor.description else []

    def initialize(self):
        self.query("""
            CREATE SCHEMA IF NOT EXISTS nebius_optuna;
            CREATE TABLE IF NOT EXISTS nebius_optuna.runs (
                run_id TEXT PRIMARY KEY, study_name TEXT UNIQUE NOT NULL,
                config JSONB NOT NULL, config_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                deadline TIMESTAMPTZ NOT NULL, stopped BOOLEAN NOT NULL DEFAULT false
            );
            CREATE TABLE IF NOT EXISTS nebius_optuna.submissions (
                submission_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES nebius_optuna.runs(run_id),
                expected_number INTEGER NOT NULL, trial_number INTEGER,
                assignment JSONB, state TEXT NOT NULL DEFAULT 'ALLOCATING',
                operation_id TEXT, job_id TEXT UNIQUE, worker_claim TEXT,
                prune_intent BOOLEAN NOT NULL DEFAULT false,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                terminal_seen_at TIMESTAMPTZ, outcome JSONB, observation JSONB,
                cancel_intent BOOLEAN NOT NULL DEFAULT false, cancel_operation TEXT,
                delete_intent BOOLEAN NOT NULL DEFAULT false, delete_operation TEXT,
                deleted BOOLEAN NOT NULL DEFAULT false, diagnostic TEXT,
                UNIQUE(run_id, trial_number)
            );
        """)

    @contextmanager
    def lock(self, name):
        key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big", signed=True)
        if not self.query("SELECT pg_try_advisory_lock(%s) AS locked", (key,))[0]["locked"]:
            raise RuntimeError("Another coordinator owns this study")
        try:
            yield
        finally:
            if not self.connection.closed:
                self.query("SELECT pg_advisory_unlock(%s)", (key,))

    def create_run(self, study_name, config, config_hash):
        run_id = str(uuid.uuid4())
        self.query(
            """INSERT INTO nebius_optuna.runs
            (run_id, study_name, config, config_hash, deadline)
            VALUES (%s, %s, %s, %s, now() + %s * interval '1 second')""",
            (run_id, study_name, Json(config), config_hash, config["run_timeout"]),
        )
        return self.run(run_id)

    def run(self, run_id):
        rows = self.query(
            "SELECT *, now() >= deadline AS expired FROM nebius_optuna.runs " "WHERE run_id=%s",
            (run_id,),
        )
        if len(rows) != 1:
            raise ValueError("Unknown run ID")
        return rows[0]

    def stop(self, run_id):
        self.query("UPDATE nebius_optuna.runs SET stopped=true WHERE run_id=%s", (run_id,))

    def rows(self, run_id):
        return self.query(
            "SELECT * FROM nebius_optuna.submissions WHERE run_id=%s "
            "ORDER BY created_at, submission_id",
            (run_id,),
        )

    def row(self, submission_id):
        rows = self.query(
            "SELECT * FROM nebius_optuna.submissions WHERE submission_id=%s", (submission_id,)
        )
        if len(rows) != 1:
            raise ValueError("Unknown submission ID")
        return rows[0]

    def allocate(self, run_id, expected_number, budget):
        # Caller holds the study lock. The INSERT also checks the total reservation budget.
        rows = self.query(
            """INSERT INTO nebius_optuna.submissions
            (submission_id, run_id, expected_number)
            SELECT %s, %s, %s WHERE
            (SELECT count(*) FROM nebius_optuna.submissions WHERE run_id=%s) < %s
            RETURNING *""",
            (str(uuid.uuid4()), run_id, expected_number, run_id, budget),
        )
        if not rows:
            raise RuntimeError("Trial allocation budget exhausted")
        return rows[0]

    def update(self, submission_id, **fields):
        allowed = {
            "trial_number",
            "assignment",
            "state",
            "operation_id",
            "job_id",
            "outcome",
            "cancel_intent",
            "cancel_operation",
            "delete_intent",
            "delete_operation",
            "deleted",
            "diagnostic",
            "prune_intent",
            "observation",
        }
        if not fields or not fields.keys() <= allowed:
            raise ValueError("Invalid journal update")
        columns = ", ".join(key + "=%s" for key in fields)
        values = [
            Json(v) if key in {"assignment", "outcome", "observation"} else v
            for key, v in fields.items()
        ]
        self.query(
            "UPDATE nebius_optuna.submissions SET " + columns + " WHERE submission_id=%s",
            (*values, submission_id),
        )

    def dispatch(self, submission_id):
        return bool(
            self.query(
                """UPDATE nebius_optuna.submissions SET state='DISPATCH_INTENT'
            WHERE submission_id=%s AND state='PREPARED' RETURNING submission_id""",
                (submission_id,),
            )
        )

    def claim(self, submission_id):
        return bool(
            self.query(
                """UPDATE nebius_optuna.submissions SET worker_claim=%s
            WHERE submission_id=%s AND worker_claim IS NULL AND outcome IS NULL
              AND state IN ('DISPATCH_INTENT','SUBMISSION_UNKNOWN','SUBMITTED')
              AND NOT cancel_intent AND NOT delete_intent
            RETURNING submission_id""",
                (str(uuid.uuid4()), submission_id),
            )
        )

    def terminal_age(self, submission_id):
        return float(
            self.query(
                """UPDATE nebius_optuna.submissions
            SET terminal_seen_at=coalesce(terminal_seen_at, now()) WHERE submission_id=%s
            RETURNING extract(epoch FROM now()-terminal_seen_at) AS age""",
                (submission_id,),
            )[0]["age"]
        )
