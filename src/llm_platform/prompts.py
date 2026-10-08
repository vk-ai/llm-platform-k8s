"""Versioned prompt registry (SQLite).

* Versions are immutable semver strings; publishing identical content is a no-op.
* Labels (``prod``, ``staging``...) point at a version; every move is recorded,
  so ``rollback`` returns a label to the version it pointed at before.
* ``diff`` is a unified diff between two versions; ``render`` fills
  ``{placeholders}`` and refuses missing or unexpected variables.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import sqlite3
import string
import threading
import time
from dataclasses import dataclass

SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class PromptError(Exception):
    pass


class NotFound(PromptError):
    pass


def parse_semver(v: str) -> tuple[int, int, int]:
    m = SEMVER.match(v)
    if not m:
        raise PromptError(f"invalid semver {v!r} (expected MAJOR.MINOR.PATCH)")
    return int(m[1]), int(m[2]), int(m[3])


def bump(v: str, part: str) -> str:
    major, minor, patch = parse_semver(v)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise PromptError(f"bump must be major|minor|patch, not {part!r}")


def variables_of(template: str) -> list[str]:
    names = []
    for _, field, _, _ in string.Formatter().parse(template):
        if field is None:
            continue
        if not field.isidentifier():
            raise PromptError(f"placeholder {{{field}}} must be a plain identifier")
        if field not in names:
            names.append(field)
    return names


@dataclass
class PromptVersion:
    name: str
    version: str
    template: str
    variables: list[str]
    sha256: str
    metadata: dict
    created_at: float

    def render(self, **values) -> str:
        missing = [v for v in self.variables if v not in values]
        extra = [k for k in values if k not in self.variables]
        if missing or extra:
            raise PromptError(f"{self.name}@{self.version}: missing {missing}, unexpected {extra}")
        return self.template.format(**values)

    def to_dict(self) -> dict:
        return {"name": self.name, "version": self.version, "template": self.template, "variables": self.variables,
                "sha256": self.sha256, "metadata": self.metadata, "created_at": self.created_at}


SCHEMA = """
CREATE TABLE IF NOT EXISTS prompt_versions (
  name TEXT NOT NULL, version TEXT NOT NULL, major INT, minor INT, patch INT,
  template TEXT NOT NULL, variables TEXT NOT NULL, sha256 TEXT NOT NULL,
  metadata TEXT NOT NULL, created_at REAL NOT NULL,
  PRIMARY KEY (name, version)
);
CREATE TABLE IF NOT EXISTS prompt_labels (
  name TEXT NOT NULL, label TEXT NOT NULL, version TEXT NOT NULL, updated_at REAL NOT NULL,
  PRIMARY KEY (name, label)
);
CREATE TABLE IF NOT EXISTS prompt_label_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, label TEXT NOT NULL,
  version TEXT NOT NULL, actor TEXT, action TEXT NOT NULL, at REAL NOT NULL
);
"""


class PromptRegistry:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _row(self, r) -> PromptVersion:
        return PromptVersion(r["name"], r["version"], r["template"], json.loads(r["variables"]), r["sha256"], json.loads(r["metadata"]), r["created_at"])

    def versions(self, name: str) -> list[PromptVersion]:
        rows = self.db.execute("SELECT * FROM prompt_versions WHERE name=? ORDER BY major, minor, patch", (name,)).fetchall()
        return [self._row(r) for r in rows]

    def names(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT DISTINCT name FROM prompt_versions ORDER BY name")]

    def latest(self, name: str) -> PromptVersion | None:
        v = self.versions(name)
        return v[-1] if v else None

    def publish(self, name: str, template: str, version: str | None = None, bump_part: str = "minor", metadata: dict | None = None) -> PromptVersion:
        variables = variables_of(template)
        digest = hashlib.sha256(template.encode()).hexdigest()
        with self.lock:
            latest = self.latest(name)
            if version is not None:
                existing = self.db.execute("SELECT sha256 FROM prompt_versions WHERE name=? AND version=?", (name, version)).fetchone()
                if existing:
                    if existing[0] == digest:
                        return self._row(self.db.execute("SELECT * FROM prompt_versions WHERE name=? AND version=?", (name, version)).fetchone())
                    raise PromptError(f"{name}@{version} already exists with different content (versions are immutable)")
            elif latest and latest.sha256 == digest:
                return latest  # identical content: no new version
            if version is None:
                version = bump(latest.version, bump_part) if latest else "1.0.0"
            new = parse_semver(version)
            if latest and new <= parse_semver(latest.version):
                raise PromptError(f"{name}: version {version} must be greater than {latest.version}")
            self.db.execute(
                "INSERT INTO prompt_versions VALUES (?,?,?,?,?,?,?,?,?,?)",
                (name, version, *new, template, json.dumps(variables), digest, json.dumps(metadata or {}), time.time()),
            )
            self.db.commit()
        return self.get(name, version=version)

    def get(self, name: str, version: str | None = None, label: str | None = None) -> PromptVersion:
        if label:
            r = self.db.execute("SELECT version FROM prompt_labels WHERE name=? AND label=?", (name, label)).fetchone()
            if not r:
                raise NotFound(f"{name}: no label {label!r}")
            version = r[0]
        if version is None:
            latest = self.latest(name)
            if not latest:
                raise NotFound(f"no prompt {name!r}")
            return latest
        r = self.db.execute("SELECT * FROM prompt_versions WHERE name=? AND version=?", (name, version)).fetchone()
        if not r:
            raise NotFound(f"{name}@{version} not found")
        return self._row(r)

    def labels(self, name: str) -> dict[str, str]:
        return {r["label"]: r["version"] for r in self.db.execute("SELECT label, version FROM prompt_labels WHERE name=? ORDER BY label", (name,))}

    def set_label(self, name: str, label: str, version: str, actor: str = "", _action: str = "set") -> None:
        self.get(name, version=version)  # must exist
        with self.lock:
            self.db.execute(
                "INSERT INTO prompt_labels VALUES (?,?,?,?) ON CONFLICT(name, label) DO UPDATE SET version=excluded.version, updated_at=excluded.updated_at",
                (name, label, version, time.time()),
            )
            self.db.execute("INSERT INTO prompt_label_history(name,label,version,actor,action,at) VALUES (?,?,?,?,?,?)", (name, label, version, actor, _action, time.time()))
            self.db.commit()

    def history(self, name: str, label: str) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT version, actor, action, at FROM prompt_label_history WHERE name=? AND label=? ORDER BY id", (name, label))]

    def rollback(self, name: str, label: str, actor: str = "") -> str:
        """Point ``label`` back at the version it held before its current one."""
        current = self.labels(name).get(label)
        if current is None:
            raise NotFound(f"{name}: no label {label!r}")
        previous = None
        for h in reversed(self.history(name, label)):
            if h["version"] != current:
                previous = h["version"]
                break
        if previous is None:
            raise PromptError(f"{name}:{label} has no earlier version to roll back to")
        self.set_label(name, label, previous, actor, _action="rollback")
        return previous

    def diff(self, name: str, a: str, b: str) -> str:
        va, vb = self.get(name, version=a), self.get(name, version=b)
        return "".join(difflib.unified_diff(
            va.template.splitlines(keepends=True), vb.template.splitlines(keepends=True),
            fromfile=f"{name}@{a}", tofile=f"{name}@{b}",
        ))

    def load_seed(self, path: str) -> None:
        """Load prompts from a YAML file: {name: [{version, template, labels: [..]}, ...]}."""
        import yaml

        data = yaml.safe_load(open(path)) or {}
        for name, versions in data.items():
            for v in versions:
                pv = self.publish(name, v["template"], version=v.get("version"), metadata=v.get("metadata"))
                for label in v.get("labels", []):
                    if self.labels(name).get(label) != pv.version:
                        self.set_label(name, label, pv.version, actor="seed")
