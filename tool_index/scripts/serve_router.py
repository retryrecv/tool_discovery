"""Launch the router service.

    uv run scripts/serve_router.py --host 0.0.0.0 --port 8080

Equivalent to::

    uvicorn tool_index.router.service:app --host ... --port ...
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    import uvicorn
    uvicorn.run(
        "tool_index.router.service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
