import hashlib
from io import BytesIO
import logging
import os
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from .class_ import Class
from .constants import (
    _MODEL_HASH_PREFIX_CHARS,
    _MODEL_HASH_TEXT_CHUNK_CHARS,
    _MODEL_LOG_PREVIEW_CHARS,
    _MODEL_PARSE_ERROR_MESSAGE,
    _XML_ATTR_EXTENDS,
    _XML_ATTR_IS_INTERFACE,
    _XML_ATTR_NAME,
    _XML_ATTR_PACKAGE,
    _XML_ATTR_REFERENCED_TYPE,
    _XML_ATTR_REVERSE_REFERENCE,
    _XML_ATTR_TYPE,
    _XML_TAG_ATTRIBUTE,
    _XML_TAG_CLASS,
    _XML_TAG_COLLECTION,
    _XML_TAG_MODEL,
    _XML_TAG_REFERENCE,
    _XML_VALUE_TRUE,
)
from .errors import ModelParseError
from .fields import Attribute, Collection, Reference

if TYPE_CHECKING:
    from .model import Model

_HTTP_SCHEMES = frozenset({"http", "https"})


def _is_http_source(source: Any) -> bool:
    try:
        parsed = urlparse(str(source))
    except Exception:
        return False
    return (parsed.scheme or "").lower() in _HTTP_SCHEMES


def _coerce_local_path(source: Any) -> Path | None:
    if isinstance(source, (bytes, bytearray)):
        try:
            source = bytes(source).decode("utf-8")
        except Exception as exc:
            raise TypeError("source bytes must be utf-8 decodable for local filesystem paths") from exc
    if isinstance(source, str):
        parsed = urlparse(source)
        if parsed.scheme and parsed.scheme.lower() not in _HTTP_SCHEMES:
            raise ValueError(f"Unsupported URL scheme for model source: {parsed.scheme!r}")
        if parsed.scheme:
            return None
        return Path(source)
    if isinstance(source, Path):
        return source
    if hasattr(source, "__fspath__"):
        return Path(source)
    return None


def _open_model_source(source: Any):
    if hasattr(source, "read"):
        return source
    if _is_http_source(source):
        from intermine314.service.transport import (
            build_session,
            is_tor_proxy_url,
            resolve_proxy_url,
        )

        proxy_url = resolve_proxy_url(None)
        session = build_session(
            proxy_url=proxy_url,
            user_agent=None,
            tor_mode=is_tor_proxy_url(proxy_url),
            strict_tor_proxy_scheme=True,
            allow_insecure_tor_proxy_scheme=False,
        )
        try:
            with session.get(str(source), timeout=None, verify=True) as response:
                response.raise_for_status()
                payload = response.content
        finally:
            session.close()
        return BytesIO(payload)

    local_path = _coerce_local_path(source)
    if local_path is not None:
        return local_path.open("rb")
    raise TypeError(
        "Unsupported model source; expected HTTP(S) URL, local filesystem path, or file-like object"
    )


def _source_ref(source: Any) -> str:
    if source is None:
        return "<none>"
    if isinstance(source, (bytes, bytearray)):
        return f"<bytes:{len(source)}>"
    if isinstance(source, str):
        stripped = source.strip()
        if stripped.startswith("<"):
            return f"<inline-xml:{len(source)} chars>"
        if len(source) > 256:
            return f"<str:{len(source)} chars>"
        return source
    name = getattr(source, "name", None)
    if isinstance(name, str) and name:
        return name
    return f"<{type(source).__name__}>"


def _truncate_preview(text: str, max_chars: int = _MODEL_LOG_PREVIEW_CHARS) -> str:
    if max_chars <= 0:
        return ""
    snippet = text[:max_chars]
    one_line = " ".join(snippet.split())
    if len(text) > max_chars:
        return one_line + "..."
    return one_line


def _hash_and_count_text_bytes(text: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    for idx in range(0, len(text), _MODEL_HASH_TEXT_CHUNK_CHARS):
        chunk = text[idx : idx + _MODEL_HASH_TEXT_CHUNK_CHARS]
        payload = chunk.encode("utf-8", errors="replace")
        digest.update(payload)
        total_bytes += len(payload)
    return digest.hexdigest(), total_bytes


def _payload_metadata(source: Any, stream: Any) -> tuple[Optional[int], Optional[str], Optional[str]]:
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        preview = _truncate_preview(data[:_MODEL_LOG_PREVIEW_CHARS].decode("utf-8", errors="replace"))
        return len(data), preview, hashlib.sha256(data).hexdigest()

    if isinstance(source, str) and source.strip().startswith("<"):
        digest, byte_count = _hash_and_count_text_bytes(source)
        return byte_count, _truncate_preview(source), digest

    getbuffer = getattr(stream, "getbuffer", None)
    if callable(getbuffer):
        try:
            buffer = getbuffer()
            byte_count = len(buffer)
            preview = _truncate_preview(bytes(buffer[:_MODEL_LOG_PREVIEW_CHARS]).decode("utf-8", errors="replace"))
            return byte_count, preview, hashlib.sha256(buffer).hexdigest()
        except Exception:
            pass

    stream_name = getattr(stream, "name", None)
    if isinstance(stream_name, str) and stream_name:
        try:
            byte_count = int(os.path.getsize(stream_name))
            return byte_count, None, None
        except Exception:
            pass

    return None, None, None


def _format_xml_parse_error(error: ET.ParseError) -> str:
    position = getattr(error, "position", None)
    if isinstance(position, tuple) and len(position) == 2:
        return f"{error}; line={position[0]} column={position[1]}"
    return str(error)


def _strip_java_prefix(type_name: str) -> str:
    value = str(type_name or "").strip()
    if not value:
        return ""
    if "." not in value:
        return value
    return value.rsplit(".", 1)[-1]


def _identity(value: str) -> str:
    return str(value)


_FIELD_SPECS = (
    (_XML_TAG_ATTRIBUTE, Attribute, _XML_ATTR_TYPE, False, _strip_java_prefix),
    (_XML_TAG_REFERENCE, Reference, _XML_ATTR_REFERENCED_TYPE, True, _identity),
    (_XML_TAG_COLLECTION, Collection, _XML_ATTR_REFERENCED_TYPE, True, _identity),
)


def parse_model_xml(model: "Model", source: Any) -> None:
    io = None
    source_ref = _source_ref(source)
    try:
        if isinstance(source, str) and source.lstrip().startswith("<"):
            io = BytesIO(source.encode("utf-8"))
        elif isinstance(source, (bytes, bytearray)) and bytes(source).lstrip().startswith(b"<"):
            io = BytesIO(bytes(source))
        else:
            io = _open_model_source(source)
        byte_count, preview, digest = _payload_metadata(source, io)
        if model.LOG.isEnabledFor(logging.DEBUG):
            digest_prefix = digest[:_MODEL_HASH_PREFIX_CHARS] if digest else None
            model.LOG.debug(
                "Parsing model XML source=%s bytes=%s sha256=%s preview=%r",
                source_ref,
                byte_count if byte_count is not None else "unknown",
                digest_prefix,
                preview,
            )
        root = ET.parse(io).getroot()
        if root.tag == _XML_TAG_MODEL:
            model_node = root
        else:
            model_node = root.find(_XML_TAG_MODEL)
        if model_node is None:
            raise ModelParseError(_MODEL_PARSE_ERROR_MESSAGE, source_ref, "No model element found")

        model.name = model_node.get(_XML_ATTR_NAME, "")
        model.package_name = model_node.get(_XML_ATTR_PACKAGE, "")
        if not model.name or not model.package_name:
            raise ModelParseError(
                _MODEL_PARSE_ERROR_MESSAGE,
                source_ref,
                "Missing required attributes in <model>: name and package",
            )

        for c in model_node.findall(_XML_TAG_CLASS):
            class_name = c.get(_XML_ATTR_NAME, "")
            if not class_name:
                raise ModelParseError(_MODEL_PARSE_ERROR_MESSAGE, source_ref, "Missing name in <class>")

            parents = [_strip_java_prefix(p) for p in c.get(_XML_ATTR_EXTENDS, "").split(" ") if len(p)]
            interface = c.get(_XML_ATTR_IS_INTERFACE, "") == _XML_VALUE_TRUE
            cl = Class(class_name, parents, model, interface)
            model.LOG.debug("Created {0}".format(cl.name))
            for tag_name, field_class, type_attr_name, has_reverse, type_normalizer in _FIELD_SPECS:
                for field_node in c.findall(tag_name):
                    field_name = field_node.get(_XML_ATTR_NAME, "")
                    if not field_name:
                        raise ModelParseError(
                            _MODEL_PARSE_ERROR_MESSAGE,
                            source_ref,
                            f"Missing name in <{tag_name}> for class {class_name}",
                        )

                    field_type_name = type_normalizer(field_node.get(type_attr_name, ""))
                    try:
                        if has_reverse:
                            reverse_name = field_node.get(_XML_ATTR_REVERSE_REFERENCE) or ""
                            field = field_class(field_name, field_type_name, cl, reverse_name)
                        else:
                            field = field_class(field_name, field_type_name, cl)
                        cl.add_field(field)
                    except ModelParseError:
                        raise
                    except Exception as field_error:
                        raise ModelParseError(
                            _MODEL_PARSE_ERROR_MESSAGE,
                            source_ref,
                            f"Failed parsing <{tag_name}> {class_name}.{field_name}: {field_error}",
                        )
                    model.LOG.debug("set %s.%s", cl.name, field.name)
            model.classes[class_name] = cl
    except ModelParseError:
        raise
    except ET.ParseError as error:
        raise ModelParseError(_MODEL_PARSE_ERROR_MESSAGE, source_ref, _format_xml_parse_error(error))
    except Exception as error:
        raise ModelParseError(_MODEL_PARSE_ERROR_MESSAGE, source_ref, error)
    finally:
        if io is not None:
            io.close()
