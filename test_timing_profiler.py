from timing_profiler import TimingProfiler


def test_disabled_timing_profiler_does_not_record_sections():
    profiler = TimingProfiler(enabled=False)

    with profiler.section("work"):
        value = 1 + 1

    assert value == 2
    assert profiler.snapshot() == {}


def test_enabled_timing_profiler_records_sections_and_prints(capsys):
    profiler = TimingProfiler(enabled=True, print_every=1, prefix="unit")

    with profiler.section("work"):
        value = 1 + 1
    profiler.maybe_print("step")

    snapshot = profiler.snapshot()
    assert value == 2
    assert snapshot["work"]["count"] == 1
    assert snapshot["work"]["total"] >= 0

    captured = capsys.readouterr()
    assert "[unit step]" in captured.out
    assert "work:" in captured.out
