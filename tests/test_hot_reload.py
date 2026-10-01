"""Alias-based, zero-downtime hot-reload verification."""

from __future__ import annotations

import dataclasses
import threading
from typing import Any, Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.model_service import ModelService, ModelValidationError
from tests.conftest import feature_record

pytestmark = pytest.mark.usefixtures("trained_client")


@pytest.fixture()
def service(trained_client: TestClient) -> Iterator[ModelService]:
    """The live service; ``@champion`` is pinned to v1 before and after every test."""
    trained_client.post("/models/versions/1/promote")
    trained_client.post("/model/reload")
    yield trained_client.app.state.services.model_service
    trained_client.post("/models/versions/1/promote")
    trained_client.post("/model/reload")


def test_reload_follows_alias_changes(trained_client: TestClient, service: ModelService) -> None:
    assert trained_client.get("/health").json()["model_version"] == 1
    assert trained_client.post("/models/versions/2/promote").status_code == 200
    # Promotion alone must NOT change what is being served ...
    assert trained_client.get("/health").json()["model_version"] == 1
    # ... only an explicit reload swaps the snapshot.
    res = trained_client.post("/model/reload").json()
    assert res == {**res, "reloaded": True, "model_version": 2, "previous_version": 1, "model_alias": "champion"}
    assert trained_client.get("/health").json()["model_version"] == 2
    assert trained_client.post("/predict", json={"features": feature_record(trained_client)}).json()["model_version"] == 2


def test_reload_by_custom_alias_and_by_version(trained_client: TestClient, service: ModelService) -> None:
    trained_client.post("/models/versions/2/alias", json={"alias": "challenger"})
    by_alias = trained_client.post("/model/reload", params={"alias": "challenger"}).json()
    assert by_alias["model_version"] == 2 and by_alias["model_alias"] == "challenger"
    by_version = trained_client.post("/model/reload", params={"version": 1}).json()
    assert by_version["model_version"] == 1 and by_version["model_alias"] is None
    trained_client.delete("/models/aliases/challenger")


def test_reload_unknown_alias_or_version_keeps_active_model(trained_client: TestClient, service: ModelService) -> None:
    assert trained_client.post("/model/reload", params={"alias": "ghost"}).status_code == 404
    assert trained_client.post("/model/reload", params={"version": 999}).status_code == 404
    assert trained_client.get("/health").json()["model_version"] == 1
    assert trained_client.post("/predict", json={"features": feature_record(trained_client)}).status_code == 200


def test_reload_history_is_recorded(trained_client: TestClient, service: ModelService) -> None:
    trained_client.post("/model/reload", params={"version": 2})
    history = trained_client.get("/model/info").json()["reload_history"]
    assert history[-1]["model_version"] == 2 and history[-1]["previous_version"] == 1
    assert history[-1]["duration_ms"] > 0


def test_failed_dry_run_leaves_active_model_untouched(
    trained_client: TestClient, service: ModelService, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Exploding:
        def predict_proba(self, _: pd.DataFrame) -> Any:
            raise RuntimeError("corrupt pipeline")

    class NaNModel:
        def predict_proba(self, X: pd.DataFrame) -> Any:
            return [[float("nan"), float("nan")]] * len(X)

    registry = service._registry
    good = registry.load_model(version=2)
    before = service.current()
    for pipeline in (Exploding(), NaNModel()):
        broken = dataclasses.replace(good, pipeline=pipeline)
        monkeypatch.setattr(registry, "load_model", lambda **_: broken)
        with pytest.raises(ModelValidationError):
            service.load(version=2)
        assert service.current() is before
    monkeypatch.undo()
    assert trained_client.post("/predict", json={"features": feature_record(trained_client)}).status_code == 200


def test_dry_run_failure_maps_to_422(trained_client: TestClient, service: ModelService,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    class Exploding:
        def predict_proba(self, _: pd.DataFrame) -> Any:
            raise RuntimeError("boom")

    good = service._registry.load_model(version=2)
    monkeypatch.setattr(service._registry, "load_model", lambda **_: dataclasses.replace(good, pipeline=Exploding()))
    r = trained_client.post("/model/reload", params={"version": 2})
    assert r.status_code == 422 and "Dry-run" in r.json()["detail"]
    assert trained_client.get("/health").json()["model_version"] == 1


def test_inflight_snapshot_survives_a_swap(service: ModelService) -> None:
    old = service.current()
    frame, _ = service.prepare([{f["name"]: f["default"] for f in _specs(service)}], old, single=True)
    service.load(version=2)
    assert service.current().version == 2
    # A request that grabbed the old snapshot before the swap still scores correctly.
    assert 0.0 <= float(service.score(old, frame)[0]) <= 1.0 and old.version == 1


def _specs(service: ModelService) -> list[dict[str, Any]]:
    from api.model_service import feature_specs

    return feature_specs(service.current())


def test_no_request_is_dropped_while_reloading(trained_client: TestClient, service: ModelService) -> None:
    """Hammer the scoring path from several threads while another thread flips v1 <-> v2."""
    errors: list[BaseException] = []
    seen: set[int] = set()
    stop = threading.Event()
    record = {f["name"]: f["default"] for f in _specs(service)}

    def reader() -> None:
        while not stop.is_set():
            try:
                model = service.current()
                frame, _ = service.prepare([record], model, single=True)
                p = service.score(model, frame)
                assert 0.0 <= float(p[0]) <= 1.0
                seen.add(model.version)
            except BaseException as exc:  # noqa: BLE001 - any failure is a dropped request
                errors.append(exc)
                return

    def swapper() -> None:
        try:
            for i in range(6):
                service.load(version=2 if i % 2 == 0 else 1)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            stop.set()

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for t in readers:
        t.start()
    swap = threading.Thread(target=swapper)
    swap.start()
    swap.join(timeout=240)
    stop.set()
    for t in readers:
        t.join(timeout=30)
    assert not errors, errors
    assert seen <= {1, 2} and len(service.history()) >= 6


def test_concurrent_reloads_are_serialised(service: ModelService) -> None:
    results: list[dict[str, Any]] = []
    threads = [threading.Thread(target=lambda v=v: results.append(service.load(version=v))) for v in (1, 2, 1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=240)
    assert len(results) == 4
    # Each event's "previous_version" must equal the version produced by the load before it.
    ordered = sorted(results, key=lambda e: e["at"])
    for earlier, later in zip(ordered, ordered[1:]):
        assert later["previous_version"] == earlier["model_version"]
