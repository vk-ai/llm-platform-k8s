import pytest

from llm_platform.prompts import NotFound, PromptError, PromptRegistry, bump, parse_semver, variables_of


def test_semver():
    assert parse_semver("1.2.3") == (1, 2, 3)
    assert bump("1.2.3", "patch") == "1.2.4" and bump("1.2.3", "minor") == "1.3.0" and bump("1.2.3", "major") == "2.0.0"
    for bad in ("1.2", "v1.2.3", "01.2.3"):
        with pytest.raises(PromptError):
            parse_semver(bad)
    with pytest.raises(PromptError):
        bump("1.0.0", "huge")


def test_variables():
    assert variables_of("Hi {name}, see {context}. {name}") == ["name", "context"]
    assert variables_of("literal {{braces}}") == []
    with pytest.raises(PromptError):
        variables_of("bad {a.b}")


def test_publish_versions_are_immutable_and_ordered():
    r = PromptRegistry()
    v1 = r.publish("p", "Hello {name}")
    assert v1.version == "1.0.0"
    assert r.publish("p", "Hello {name}").version == "1.0.0"  # identical content: no-op
    assert r.publish("p", "Hi {name}").version == "1.1.0"
    assert r.publish("p", "Hey {name}", bump_part="major").version == "2.0.0"
    assert r.publish("p", "Yo {name}", version="2.0.5").version == "2.0.5"
    with pytest.raises(PromptError):
        r.publish("p", "Sup {name}", version="2.0.1")  # must increase
    assert [v.version for v in r.versions("p")] == ["1.0.0", "1.1.0", "2.0.0", "2.0.5"]
    assert r.get("p").version == "2.0.5"
    assert r.get("p", version="1.0.0").template == "Hello {name}"


def test_render_strict():
    r = PromptRegistry()
    v = r.publish("p", "Q: {question}\nC: {context}")
    assert v.render(question="q", context="c") == "Q: q\nC: c"
    with pytest.raises(PromptError):
        v.render(question="q")
    with pytest.raises(PromptError):
        v.render(question="q", context="c", extra=1)


def test_labels_history_and_rollback():
    r = PromptRegistry()
    for t in ("a {x}", "b {x}", "c {x}"):
        r.publish("p", t)
    r.set_label("p", "prod", "1.0.0", actor="alice")
    r.set_label("p", "prod", "1.1.0", actor="alice")
    r.set_label("p", "prod", "1.2.0", actor="bob")
    assert r.get("p", label="prod").version == "1.2.0"
    assert r.rollback("p", "prod", actor="oncall") == "1.1.0"
    assert r.get("p", label="prod").version == "1.1.0"
    assert [h["action"] for h in r.history("p", "prod")] == ["set", "set", "set", "rollback"]
    with pytest.raises(NotFound):
        r.set_label("p", "prod", "9.9.9")
    with pytest.raises(NotFound):
        r.get("p", label="staging")


def test_rollback_needs_history():
    r = PromptRegistry()
    r.publish("p", "a {x}")
    r.set_label("p", "prod", "1.0.0")
    with pytest.raises(PromptError):
        r.rollback("p", "prod")
    with pytest.raises(NotFound):
        r.rollback("p", "canary")


def test_diff():
    r = PromptRegistry()
    r.publish("p", "line one\nanswer briefly\n")
    r.publish("p", "line one\nanswer in bullet points\n")
    d = r.diff("p", "1.0.0", "1.1.0")
    assert "--- p@1.0.0" in d and "-answer briefly" in d and "+answer in bullet points" in d


def test_persistence_and_seed(tmp_path):
    db = str(tmp_path / "reg.sqlite")
    seed = tmp_path / "seed.yaml"
    seed.write_text("greet:\n  - version: 1.0.0\n    template: 'Hi {name}'\n    labels: [prod]\n  - version: 1.1.0\n    template: 'Hello {name}'\n")
    r = PromptRegistry(db)
    r.load_seed(str(seed))
    r.load_seed(str(seed))  # idempotent
    r2 = PromptRegistry(db)
    assert [v.version for v in r2.versions("greet")] == ["1.0.0", "1.1.0"]
    assert r2.labels("greet") == {"prod": "1.0.0"}
    assert len(r2.history("greet", "prod")) == 1
