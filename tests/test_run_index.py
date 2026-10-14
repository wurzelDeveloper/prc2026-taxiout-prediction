"""`prc.tracking`'s shared files: ``runs/index.csv`` appended under a lock and moved into place whole (a GPU chain and
`prc predict` finishing together must not drop each other's row), a crashed writer's lock taken over, and run ids that
never collide within one second. No competition data."""

from __future__ import annotations

import os
import threading
import time

import pandas as pd
import pytest
import synthetic_tree as st

from prc import tracking
from prc.config import load_config


def test_concurrent_appends_keep_every_row(tmp_path):
    index = tmp_path / "index.csv"
    errors = []

    def writer(k: int) -> None:
        try:
            for i in range(15):
                tracking._append_csv(index, pd.DataFrame([{"run_id": f"w{k}_{i}", f"col_{k}": i}]))
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    frame = pd.read_csv(index)
    assert len(frame) == 60 and frame["run_id"].is_unique
    assert {f"col_{k}" for k in range(4)} <= set(frame.columns)            # the header widened as columns appeared
    assert not list(tmp_path.glob("*.lock")) and not list(tmp_path.glob("*.tmp"))


def test_a_crashed_writers_lock_is_taken_over(tmp_path, monkeypatch):
    index = tmp_path / "index.csv"
    lock = tmp_path / "index.csv.lock"
    lock.write_text("12345", encoding="ascii")
    old = time.time() - tracking.LOCK_STALE_S - 5
    os.utime(lock, (old, old))
    tracking._append_csv(index, pd.DataFrame([{"run_id": "a"}]))
    assert pd.read_csv(index)["run_id"].tolist() == ["a"] and not lock.exists()

    lock.write_text("12345", encoding="ascii")                               # a live writer: wait, then give up
    monkeypatch.setattr(tracking, "LOCK_TIMEOUT_S", 0.3)
    with pytest.raises(TimeoutError, match="held by another writer"):
        tracking._append_csv(index, pd.DataFrame([{"run_id": "b"}]))
    assert pd.read_csv(index)["run_id"].tolist() == ["a"]


def test_two_runs_of_one_config_in_the_same_second_get_their_own_directories(tmp_path, monkeypatch):
    st.retarget_paths(monkeypatch, tmp_path / "data", tmp_path)
    cfg = load_config("base.yaml").replace(name="twice")
    first = tracking.Run(cfg, seed=1)
    second = tracking.Run(cfg, seed=1)
    assert first.run_id != second.run_id and first.dir.exists() and second.dir.exists()
    given = tracking.Run(cfg, run_id=first.run_id)                          # an id given explicitly is used as given
    assert given.run_id == first.run_id
