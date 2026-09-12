from __future__ import annotations

import base64
import copy
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import unquote, urlsplit

from .artifacts import MAX_ARTIFACT_BYTES, ArtifactService
from .attachments import (
    HostResolvedAttachment,
    normalize_host_resolved_attachments,
)


_MULTIMODAL_PROVIDERS = frozenset(
    {"openai", "anthropic", "hyperspace", "ollama", "gemini"}
)
_PROVENANCE_ONLY_ATTACHMENT_KINDS = frozenset({"handoff"})


class ContextModelProjectionError(RuntimeError):
    """A durable journal attachment could not become safe model input."""

    code = "context_v2_model_projection_invalid"
    boundary = "model_context_projection"

    def __init__(
        self,
        reason: str,
        *,
        message_index: int | None = None,
        attachment_index: int | None = None,
    ) -> None:
        self.reason = str(reason or "invalid")[:128]
        self.message_index = message_index
        self.attachment_index = attachment_index
        super().__init__(self.code)


RemoteAttachmentSourceDecoder = Callable[
    [HostResolvedAttachment, bytes],
    Mapping[str, Any],
]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_plain(item) for item in value]
    return copy.deepcopy(value)


def _canonical_remote_source(value: Mapping[str, Any]) -> dict[str, str]:
    source = _plain(value)
    source_type = source.get("type")
    if source_type == "url":
        if not set(source).issubset({"type", "url", "media_type"}):
            raise ContextModelProjectionError("remote_source_fields_invalid")
        url = source.get("url")
        if (
            not isinstance(url, str)
            or not url
            or url != url.strip()
            or len(url) > 8_192
            or "\\" in url
            or any(
                character.isspace()
                or unicodedata.category(character).startswith("C")
                for character in url
            )
        ):
            raise ContextModelProjectionError("remote_source_url_invalid")
        hexadecimal = frozenset("0123456789abcdefABCDEF")
        if any(
            character == "%"
            and (
                index + 2 >= len(url)
                or url[index + 1] not in hexadecimal
                or url[index + 2] not in hexadecimal
            )
            for index, character in enumerate(url)
        ):
            raise ContextModelProjectionError("remote_source_url_invalid")
        try:
            decoded_url = unquote(url, encoding="utf-8", errors="strict")
            parsed_urls = (urlsplit(url), urlsplit(decoded_url))
            parsed_ports = tuple(candidate.port for candidate in parsed_urls)
        except (UnicodeDecodeError, ValueError) as error:
            raise ContextModelProjectionError(
                "remote_source_url_invalid"
            ) from error
        if (
            "\\" in decoded_url
            or any(
                character.isspace()
                or unicodedata.category(character).startswith("C")
                for character in decoded_url
            )
            or any("%" in candidate.netloc for candidate in parsed_urls)
            or any(
                candidate.scheme.casefold() not in {"http", "https"}
                or not candidate.netloc
                or not candidate.hostname
                or candidate.username is not None
                or candidate.password is not None
                for candidate in parsed_urls
            )
            or any(
                port is not None and not 1 <= port <= 65_535
                for port in parsed_ports
            )
        ):
            raise ContextModelProjectionError("remote_source_url_invalid")
        result = {"type": "url", "url": url}
        media_type = source.get("media_type")
        if media_type is not None:
            if (
                not isinstance(media_type, str)
                or not media_type
                or media_type != media_type.strip()
                or len(media_type) > 255
                or "/" not in media_type
                or any(ord(character) < 32 for character in media_type)
            ):
                raise ContextModelProjectionError(
                    "remote_source_media_type_invalid"
                )
            result["media_type"] = media_type.casefold()
        return result
    if source_type == "file_id":
        if set(source) != {"type", "file_id"}:
            raise ContextModelProjectionError("remote_source_fields_invalid")
        file_id = source.get("file_id")
        if (
            not isinstance(file_id, str)
            or not file_id
            or file_id != file_id.strip()
            or len(file_id) > 4_096
            or any(ord(character) < 32 for character in file_id)
        ):
            raise ContextModelProjectionError("remote_source_file_id_invalid")
        return {"type": "file_id", "file_id": file_id}
    raise ContextModelProjectionError("remote_source_type_unsupported")


def _attachment_modality(attachment: HostResolvedAttachment) -> str | None:
    kind = attachment.kind.casefold()
    media_type = attachment.media_type.split(";", 1)[0].strip().casefold()
    if kind == "image":
        return "image"
    if kind == "pdf" or (kind == "file" and media_type == "application/pdf"):
        return "pdf"
    if kind in _PROVENANCE_ONLY_ATTACHMENT_KINDS:
        return None
    raise ContextModelProjectionError("attachment_kind_unsupported")


class ModelContextProjection:
    """Project journal-safe attachment envelopes into canonical model blocks."""

    def __init__(
        self,
        artifacts: ArtifactService,
        *,
        remote_source_decoder: RemoteAttachmentSourceDecoder | None = None,
    ) -> None:
        if not isinstance(artifacts, ArtifactService):
            raise TypeError("artifacts must be an ArtifactService")
        if remote_source_decoder is not None and not callable(
            remote_source_decoder
        ):
            raise TypeError("remote_source_decoder must be callable")
        self._artifacts = artifacts
        self._remote_source_decoder = remote_source_decoder

    @property
    def artifacts(self) -> ArtifactService:
        return self._artifacts

    def _source(
        self,
        attachment: HostResolvedAttachment,
        *,
        remaining_budget_bytes: int,
    ) -> tuple[dict[str, str], int]:
        try:
            content = self._artifacts.read_full(
                attachment.artifact,
                remaining_budget_bytes=remaining_budget_bytes,
            )
        except Exception as error:
            raise ContextModelProjectionError(
                "attachment_artifact_unavailable"
            ) from error
        media_type = attachment.media_type.split(";", 1)[0].strip().casefold()
        if media_type.startswith("image/") or media_type == "application/pdf":
            source = {
                "type": "base64",
                "media_type": media_type,
                "data": base64.b64encode(content).decode("ascii"),
            }
            return source, len(content)
        if self._remote_source_decoder is None:
            raise ContextModelProjectionError(
                "remote_attachment_decoder_unavailable"
            )
        try:
            decoded = self._remote_source_decoder(attachment, content)
        except ContextModelProjectionError:
            raise
        except Exception as error:
            raise ContextModelProjectionError(
                "remote_attachment_descriptor_invalid"
            ) from error
        if not isinstance(decoded, Mapping):
            raise ContextModelProjectionError(
                "remote_attachment_descriptor_invalid"
            )
        return _canonical_remote_source(decoded), len(content)

    def _attachment_block(
        self,
        attachment: HostResolvedAttachment,
        *,
        remaining_budget_bytes: int,
    ) -> tuple[dict[str, Any] | None, int]:
        modality = _attachment_modality(attachment)
        if modality is None:
            return None, 0
        source, consumed_bytes = self._source(
            attachment,
            remaining_budget_bytes=remaining_budget_bytes,
        )
        if modality == "image":
            source_media_type = source.get("media_type")
            if source["type"] == "base64" and not str(
                source_media_type or ""
            ).startswith("image/"):
                raise ContextModelProjectionError(
                    "attachment_media_type_mismatch"
                )
            if (
                source["type"] == "file_id"
                or (
                    isinstance(source_media_type, str)
                    and not source_media_type.startswith("image/")
                )
            ):
                raise ContextModelProjectionError(
                    "attachment_media_type_mismatch"
                )
            return {"type": "image", "source": source}, consumed_bytes
        if source["type"] in {"file_id", "url"}:
            if source["type"] == "url" and source.get("media_type") not in (
                None,
                "application/pdf",
            ):
                raise ContextModelProjectionError(
                    "attachment_media_type_mismatch"
                )
            return {"type": "pdf", "source": source}, consumed_bytes
        if source.get("media_type") != "application/pdf":
            raise ContextModelProjectionError("attachment_media_type_mismatch")
        source["filename"] = attachment.name
        return {"type": "pdf", "source": source}, consumed_bytes

    def project(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        provider: str,
    ) -> tuple[dict[str, Any], ...]:
        if not isinstance(messages, Sequence) or isinstance(
            messages,
            (str, bytes, bytearray),
        ):
            raise TypeError("messages must be an array")
        normalized_provider = str(provider or "").strip().casefold()
        projected: list[dict[str, Any]] = []
        consumed_bytes = 0
        for message_index, raw_message in enumerate(messages):
            if not isinstance(raw_message, Mapping):
                raise ContextModelProjectionError(
                    "message_shape_invalid",
                    message_index=message_index,
                )
            message = _plain(raw_message)
            raw_attachments = message.pop("attachments", None)
            if raw_attachments is None:
                projected.append(message)
                continue
            try:
                attachments = normalize_host_resolved_attachments(
                    raw_attachments
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ContextModelProjectionError(
                    "attachment_envelope_invalid",
                    message_index=message_index,
                ) from error
            if message.get("role") != "user":
                raise ContextModelProjectionError(
                    "attachment_role_invalid",
                    message_index=message_index,
                )
            blocks: list[dict[str, Any]] = []
            for attachment_index, attachment in enumerate(attachments):
                try:
                    block, used_bytes = self._attachment_block(
                        attachment,
                        remaining_budget_bytes=MAX_ARTIFACT_BYTES - consumed_bytes,
                    )
                except ContextModelProjectionError as error:
                    if error.message_index is None:
                        error.message_index = message_index
                    if error.attachment_index is None:
                        error.attachment_index = attachment_index
                    raise
                consumed_bytes += used_bytes
                if block is not None:
                    blocks.append(block)
            if blocks and normalized_provider not in _MULTIMODAL_PROVIDERS:
                raise ContextModelProjectionError(
                    "provider_multimodal_unsupported",
                    message_index=message_index,
                )
            if blocks and normalized_provider == "ollama" and any(
                block.get("type") != "image"
                or not isinstance(block.get("source"), Mapping)
                or block["source"].get("type") != "base64"
                for block in blocks
            ):
                raise ContextModelProjectionError(
                    "provider_attachment_unsupported",
                    message_index=message_index,
                )
            if not blocks:
                projected.append(message)
                continue
            content = message.get("content")
            if isinstance(content, str):
                canonical_content: list[Any] = []
                if content:
                    canonical_content.append({"type": "text", "text": content})
            elif isinstance(content, Sequence) and not isinstance(
                content,
                (str, bytes, bytearray),
            ):
                canonical_content = _plain(content)
            else:
                raise ContextModelProjectionError(
                    "attachment_content_invalid",
                    message_index=message_index,
                )
            message["content"] = [*canonical_content, *blocks]
            projected.append(message)
        return tuple(projected)


__all__ = [
    "ContextModelProjectionError",
    "ModelContextProjection",
    "RemoteAttachmentSourceDecoder",
]
