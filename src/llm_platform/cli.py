"""`llm-platform` CLI: serve the gateway, run a mock model server, manage prompts."""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="llm-platform")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    s = sub.add_parser("mock-backend")
    s.add_argument("--name", default=os.environ.get("MOCK_NAME", "mock"))
    s.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("prompts")
    p.add_argument("--db", default=os.environ.get("LLMPLAT_REGISTRY", "prompts.sqlite"))
    ps = p.add_subparsers(dest="action", required=True)
    x = ps.add_parser("publish"); x.add_argument("name"); x.add_argument("file"); x.add_argument("--version"); x.add_argument("--bump", default="minor")
    x = ps.add_parser("list"); x.add_argument("name", nargs="?")
    x = ps.add_parser("show"); x.add_argument("name"); x.add_argument("--version"); x.add_argument("--label")
    x = ps.add_parser("diff"); x.add_argument("name"); x.add_argument("a"); x.add_argument("b")
    x = ps.add_parser("label"); x.add_argument("name"); x.add_argument("label"); x.add_argument("version")
    x = ps.add_parser("rollback"); x.add_argument("name"); x.add_argument("label")
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn

        from .app import create_app

        uvicorn.run(create_app(), host=a.host, port=a.port)
        return 0
    if a.cmd == "mock-backend":
        import uvicorn

        from .mock_server import create_mock_app

        uvicorn.run(create_mock_app(a.name), host="0.0.0.0", port=a.port)
        return 0

    from .prompts import PromptError, PromptRegistry

    reg = PromptRegistry(a.db)
    try:
        if a.action == "publish":
            v = reg.publish(a.name, open(a.file).read(), a.version, a.bump)
            print(f"{a.name}@{v.version} sha256={v.sha256[:12]} variables={v.variables}")
        elif a.action == "list":
            for n in [a.name] if a.name else reg.names():
                print(json.dumps({"name": n, "versions": [v.version for v in reg.versions(n)], "labels": reg.labels(n)}))
        elif a.action == "show":
            print(json.dumps(reg.get(a.name, a.version, a.label).to_dict(), indent=2))
        elif a.action == "diff":
            print(reg.diff(a.name, a.a, a.b), end="")
        elif a.action == "label":
            reg.set_label(a.name, a.label, a.version, actor=os.environ.get("USER", "cli"))
            print(json.dumps(reg.labels(a.name)))
        elif a.action == "rollback":
            print(f"{a.name}:{a.label} -> {reg.rollback(a.name, a.label, actor=os.environ.get('USER', 'cli'))}")
    except PromptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
