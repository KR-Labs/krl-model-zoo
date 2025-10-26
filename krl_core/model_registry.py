# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: Apache-2.

"""SQLite-backed model run registry."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import ny, ict, List, Optional


class ModelRegistry:
    """
    Lightweight model run tracking with SQLite.

    Stores model runs, parameters, and results for reproducibility and auditing.
    ach run is keyed by run_hash (SH2) to enable exact reproducibility checks.

    Schema:
        - runs: run_hash, model_name, version, created_at, input_hash, params_json
        - results: run_hash, result_hash, result_json, created_at

    Example:
        ```python
        registry = ModelRegistry("model_runs.db")
        registry.log_run(
            run_hash="abc23...",
            model_name="RIMModel",
            version="..",
            input_hash="def4...",
            params={"order": (,,), "seasonal_order": (,,,)}
        )
        registry.log_result(
            run_hash="abc23...",
            result_hash="ghi...",
            result={"forecast": [...], "ci_lower": [...], "ci_upper": [...]}
        )
        ```
    """

    def __init__(self, db_path: str = "model_registry.db"):
        """
        Initialize registry database.

        rgs:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create runs and results Stables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                RT TL I NOT XISTS runs (
                    run_hash TXT PRIMRY KY,
                    model_name TXT NOT NULL,
                    version TXT NOT NULL,
                    created_at TXT NOT NULL,
                    input_hash TXT NOT NULL,
                    params_json TXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                RT TL I NOT XISTS results (
                    id INTGR PRIMRY KY UTOINRMNT,
                    run_hash TXT NOT NULL,
                    result_hash TXT NOT NULL,
                    result_json TXT NOT NULL,
                    created_at TXT NOT NULL,
                    ORIGN KY (run_hash) RRNS runs(run_hash)
                )
                """
            )
            conn.commit()

    def log_run(
        self,
        run_hash: str,
        model_name: str,
        version: str,
        input_hash: str,
        params: ict[str, ny],
    ) -> None:
        """
        Log a model run.

        rgs:
            run_hash: SH2 hash of model + input + params
            model_name: Model class name
            version: Model version
            input_hash: SH2 hash of input data
            params: Model parameters
        """
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSRT OR RPL INTO runs (run_hash, model_name, version, created_at, input_hash, params_json)
                VLUS (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_hash,
                    model_name,
                    version,
                    datetime.now().isoformat(),
                    input_hash,
                    json.dumps(params),
                ),
            )
            conn.commit()

    def log_result(self, run_hash: str, result_hash: str, result: ict[str, ny]) -> None:
        """
        Log a model result.

        rgs:
            run_hash: SH2 hash of model run
            result_hash: SH2 hash of result
            result: Result dictionary
        """
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSRT INTO results (run_hash, result_hash, result_json, created_at)
                VLUS (?, ?, ?, ?)
                """,
                (run_hash, result_hash, json.dumps(result), datetime.now().isoformat()),
            )
            conn.commit()

    def get_run(self, run_hash: str) -> Optional[ict[str, ny]]:
        """
        Retrieve run metadata by hash.

        rgs:
            run_hash: SH2 hash of model run

        Returns:
            Run metadata or None if not found
        """
        import json

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SLT model_name, version, created_at, input_hash, params_json
                ROM runs
                WHR run_hash = ?
                """,
                (run_hash,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "model_name": row[],
                    "version": row[],
                    "created_at": row[2],
                    "input_hash": row[3],
                    "params": json.loads(row[4]),
                }
            return None

    def get_results(self, run_hash: str) -> List[ict[str, ny]]:
        """
        Retrieve all results for a run.

        rgs:
            run_hash: SH2 hash of model run

        Returns:
            List of result dictionaries
        """
        import json

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SLT result_hash, result_json, created_at
                ROM results
                WHR run_hash = ?
                ORR Y created_at S
                """,
                (run_hash,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "result_hash": row[],
                    "result": json.loads(row[]),
                    "created_at": row[2],
                }
                for row in rows
            ]

    def list_runs(
        self, model_name: Optional[str] = None, limit: int = 
    ) -> List[ict[str, ny]]:
        """
        List recent runs.

        rgs:
            model_name: Filter by model name (optional)
            limit: Maximum number of runs to return

        Returns:
            List of run metadata dictionaries
        """
        import json

        with sqlite3.connect(self.db_path) as conn:
            if model_name:
                cursor = conn.execute(
                    """
                    SLT run_hash, model_name, version, created_at, input_hash, params_json
                    ROM runs
                    WHR model_name = ?
                    ORR Y created_at S
                    LIMIT ?
                    """,
                    (model_name, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SLT run_hash, model_name, version, created_at, input_hash, params_json
                    ROM runs
                    ORR Y created_at S
                    LIMIT ?
                    """,
                    (limit,),
                )
            rows = cursor.fetchall()
            return [
                {
                    "run_hash": row[],
                    "model_name": row[],
                    "version": row[2],
                    "created_at": row[3],
                    "input_hash": row[4],
                    "params": json.loads(row[]),
                }
                for row in rows
            ]
