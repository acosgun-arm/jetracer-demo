#!/usr/bin/env python3
"""Serve the lock-safe browser lane annotation UI for one video workspace."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import unquote, urlparse

import jetracer_sim as sim


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "examples" / "lane_annotation.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "video_lane_calibration.json",
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    arguments = parser.parse_args()
    config = sim.load_video_lane_calibration_config(arguments.config)
    workspace = arguments.workspace.resolve()
    if not (workspace / "workspace.json").is_file():
        raise SystemExit(f"workspace.json not found under {workspace}")
    host = arguments.host or str(config["server"]["host"])
    port = arguments.port or int(config["server"]["port"])
    handler = _handler(
        workspace,
        int(config["annotations"]["maximum_request_bytes"]),
        config,
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"annotation_url=http://{host}:{port}")
    print("No native camera or GUI window is created.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _handler(
    workspace: Path,
    maximum_request_bytes: int,
    config: dict[str, Any],
) -> type[BaseHTTPRequestHandler]:
    lock = Lock()
    workspace_file = workspace / "workspace.json"
    annotations_file = workspace / "annotations.json"
    frames_root = (workspace / "frames").resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send_bytes(HTML.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/session":
                with lock:
                    workspace_document = json.loads(
                        workspace_file.read_text(encoding="utf-8")
                    )
                    annotation_document = sim.load_video_lane_annotations(
                        annotations_file
                    )
                    document = {
                        "workspace": workspace_document,
                        "annotations": annotation_document,
                        "uncertainty_review": sim.rank_video_lane_review_frames(
                            workspace_document, annotation_document, config
                        ),
                    }
                self._send_json(document)
                return
            if path.startswith("/frames/"):
                name = unquote(path.removeprefix("/frames/"))
                candidate = (frames_root / name).resolve()
                if candidate.parent != frames_root or not candidate.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send_bytes(candidate.read_bytes(), "image/png")
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/annotations":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > maximum_request_bytes:
                    raise ValueError("invalid annotation request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("annotation request must be an object")
                frame_id = payload.get("frame_id")
                annotation = payload.get("annotation")
                workspace_document = json.loads(
                    workspace_file.read_text(encoding="utf-8")
                )
                known = {frame["frame_id"] for frame in workspace_document["frames"]}
                if frame_id not in known or not isinstance(annotation, dict):
                    raise ValueError("unknown frame or invalid annotation")
                with lock:
                    document = sim.load_video_lane_annotations(annotations_file)
                    updated = json.loads(json.dumps(document))
                    updated["frames"][str(frame_id)] = annotation
                    sim.save_video_lane_annotations(annotations_file, updated)
                self._send_json({"saved": True, "frame_id": frame_id})
            except (json.JSONDecodeError, OSError, ValueError) as error:
                self._send_json({"saved": False, "error": str(error)}, status=400)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"annotation_http: {format % args}")

        def _send_json(self, document: Any, *, status: int = 200) -> None:
            self._send_bytes(
                json.dumps(document).encode("utf-8"),
                "application/json",
                status=status,
            )

        def _send_bytes(self, content: bytes, content_type: str, *, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

    return Handler


if __name__ == "__main__":
    main()
