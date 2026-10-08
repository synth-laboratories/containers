from types import SimpleNamespace

from synth_containers import pid1


def test_dead_engine_restarts_on_same_port_before_facade_exits(monkeypatch) -> None:
    spec = pid1.ChildProcess(name="engine", argv=["engine"], port=None)
    stopped = []
    original = pid1.StartedChild(
        name="engine",
        url="http://127.0.0.1:43123",
        port=43123,
        process=SimpleNamespace(returncode=7, poll=lambda: 7),
        stop=lambda: stopped.append("original"),
    )
    restarted = pid1.StartedChild(
        name="engine",
        url="http://127.0.0.1:43123",
        port=43123,
        process=SimpleNamespace(returncode=8, poll=lambda: 8),
        stop=lambda: stopped.append("restarted"),
    )
    observed = []

    def restart(pinned):
        observed.append(pinned)
        return restarted

    monkeypatch.setenv("SYNTH_CHILD_RESTART_LIMIT", "1")
    monkeypatch.setattr(pid1, "start_child", restart)

    started = [original]
    pid1._block_until_signal(object(), started, [spec])

    assert len(observed) == 1
    assert observed[0].port == 43123
    assert started == [restarted]
    assert stopped == []
