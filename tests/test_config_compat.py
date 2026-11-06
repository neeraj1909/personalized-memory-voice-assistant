from personalized_memory.compat import getenv


def test_new_configuration_takes_precedence_and_legacy_remains_supported(monkeypatch):
    monkeypatch.setenv("ASTRO_MEMORY_FAST", "old")
    monkeypatch.delenv("PMVA_MEMORY_FAST", raising=False)
    assert getenv("PMVA_MEMORY_FAST") == "old"
    monkeypatch.setenv("PMVA_MEMORY_FAST", "new")
    assert getenv("PMVA_MEMORY_FAST") == "new"
