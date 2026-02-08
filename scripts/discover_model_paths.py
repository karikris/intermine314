#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from intermine314.webservice import Service
from scripts.bench_utils import parse_csv_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect InterMine model classes/fields for safe query path discovery."
    )
    parser.add_argument(
        "--mine-url",
        required=True,
        help="InterMine mine URL or /service root (for example: https://bar.utoronto.ca/thalemine).",
    )
    parser.add_argument(
        "--classes",
        default="Gene,Transcript,CDS,Protein",
        help="Comma-separated class names to inspect.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path for JSON report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    classes = parse_csv_tokens(args.classes)

    service = Service(args.mine_url)
    model = service.model

    payload = {
        "mine_url": args.mine_url,
        "service_version": service.version,
        "service_release": service.release,
        "classes": {},
    }
    for class_name in classes:
        try:
            cls = model.get_class(class_name)
        except Exception as exc:
            payload["classes"][class_name] = {"error": str(exc)}
            continue
        fields = []
        for field_name in sorted(cls.field_dict.keys()):
            field = cls.get_field(field_name)
            entry = {
                "name": field.name,
                "kind": field.__class__.__name__,
            }
            if hasattr(field, "type_name"):
                entry["type_name"] = getattr(field, "type_name")
            if hasattr(field, "type_class") and getattr(field, "type_class") is not None:
                entry["type_class"] = getattr(field.type_class, "name", None)
            fields.append(entry)
        payload["classes"][class_name] = {
            "field_count": len(fields),
            "fields": fields,
        }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
