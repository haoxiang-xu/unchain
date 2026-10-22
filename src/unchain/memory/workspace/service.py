from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Callable, Sequence

from unchain.journal import ModelValidationError, OperationRef, ResourceRef
from unchain.journal.models import _required_text

from .models import (
    MemoryEntry,
    MemoryEntryKind,
    MemoryEntryPage,
    MemoryLink,
    MemorySpace,
    canonical_memory_tags,
    canonical_memory_link_url,
)
from .operations import build_operation_ref
from .paths import (
    canonical_entry_path,
    canonical_parent_path,
    meaningful_description,
    virtual_name,
)
from .ports import (
    BoundMemoryWorkspaceRepository,
    BoundWorkspaceContentRepository,
    BoundWorkspaceHistoryRepository,
    BoundWorkspaceLinkRepository,
    BoundWorkspaceMutationRepository,
    BoundWorkspaceReferenceAuthorizer,
    RepositoryConflictError,
    RepositoryNotFoundError,
    RepositoryScopeError,
    WorkspaceContentPage,
    WorkspaceLinkRequest,
    WorkspaceMutationRequest,
    WorkspaceRepositoryError,
)
from .search import WorkspaceSearchResult, WorkspaceSearchService


MAX_LIST_RESULTS = 200
MAX_LIST_SCAN_ENTRIES = 10_000
MAX_READ_BYTES = 32 * 1024
MAX_WRITE_BYTES = 256 * 1024
MAX_HISTORY_REVISIONS = 50
_SOURCE_KINDS = frozenset(
    {"artifact", "checkpoint", "context_event", "handoff", "memory"}
)


@dataclass(frozen=True)
class WorkspaceWriteDraft:
    """Immutable, normalized write data exposed to a host redactor."""

    path: str
    description: str
    kind: MemoryEntryKind
    content: bytes | None
    media_type: str
    link_url: str
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise TypeError("path must be text")
        if not isinstance(self.description, str):
            raise TypeError("description must be text")
        if not isinstance(self.kind, MemoryEntryKind):
            raise TypeError("kind must be a MemoryEntryKind")
        if self.content is not None and not isinstance(self.content, bytes):
            raise TypeError("content must be bytes")
        if not isinstance(self.media_type, str):
            raise TypeError("media_type must be text")
        if not isinstance(self.link_url, str):
            raise TypeError("link_url must be text")
        if not isinstance(self.tags, tuple) or any(
            not isinstance(item, str) for item in self.tags
        ):
            raise TypeError("tags must be a tuple of text values")


def _identity_content_redactor(draft: WorkspaceWriteDraft) -> WorkspaceWriteDraft:
    return draft


def _bounded_limit(value: object, *, maximum: int, field_name: str = "limit") -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > maximum
    ):
        raise ValueError(f"{field_name} must be between 1 and {maximum}")
    return value


def _bounded_offset(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("offset must be a non-negative integer")
    return value


def _positive_revision(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _stable_id(prefix: str, binding: str, operation: OperationRef) -> str:
    digest = hashlib.sha256(
        f"{binding}\0{operation.operation_id}".encode("utf-8")
    ).hexdigest()
    return f"{prefix}-{digest[:32]}"


class _BoundWorkspaceReads:
    def __init__(
        self,
        *,
        repository: BoundMemoryWorkspaceRepository,
        content: BoundWorkspaceContentRepository,
        history: BoundWorkspaceHistoryRepository,
        search: WorkspaceSearchService | None = None,
    ) -> None:
        if not isinstance(repository, BoundMemoryWorkspaceRepository):
            raise TypeError("repository must be a BoundMemoryWorkspaceRepository")
        if not isinstance(content, BoundWorkspaceContentRepository):
            raise TypeError("content must be a BoundWorkspaceContentRepository")
        if not isinstance(history, BoundWorkspaceHistoryRepository):
            raise TypeError("history must be a BoundWorkspaceHistoryRepository")
        self._repository = repository
        self._content = content
        self._history = history
        self._require_same_space(content, "content")
        self._require_same_space(history, "history")
        if search is not None and search.repository.space.space_id != self.space.space_id:
            raise RepositoryScopeError("search belongs to another workspace")
        self._search = search or WorkspaceSearchService(repository=repository)

    @property
    def repository(self) -> BoundMemoryWorkspaceRepository:
        return self._repository

    @property
    def space(self) -> MemorySpace:
        return self._repository.space

    def list(
        self,
        *,
        parent_path: str = "/",
        include_deleted: bool = False,
        recursive: bool = True,
        limit: int = 100,
        cursor: str | None = None,
    ) -> MemoryEntryPage:
        path = canonical_parent_path(parent_path)
        if not isinstance(recursive, bool):
            raise TypeError("recursive must be a boolean")
        page_limit = _bounded_limit(limit, maximum=MAX_LIST_RESULTS)
        matched: list[MemoryEntry] = []
        repository_cursor: str | None = None
        seen_cursors: set[str] = set()
        scanned_count = 0
        while scanned_count < MAX_LIST_SCAN_ENTRIES:
            repository_page = self._repository.list_entries(
                parent_path=path,
                include_deleted=include_deleted,
                limit=min(MAX_LIST_RESULTS, MAX_LIST_SCAN_ENTRIES - scanned_count),
                cursor=repository_cursor,
            )
            scanned_count += len(repository_page.entries)
            for entry in repository_page.entries:
                self._validate_entry(entry)
                prefix = path.rstrip("/") + "/"
                if path != "/" and not entry.path.startswith(prefix):
                    raise RepositoryScopeError(
                        "repository returned an entry outside the requested path"
                    )
                if entry.deleted and not include_deleted:
                    raise WorkspaceRepositoryError("repository returned an archived entry")
                parent = entry.path.rsplit("/", 1)[0] or "/"
                if recursive or parent == path:
                    matched.append(entry)
            if not repository_page.has_more:
                break
            next_cursor = repository_page.next_cursor
            if next_cursor is None or next_cursor in seen_cursors:
                raise WorkspaceRepositoryError("workspace pagination did not advance")
            if scanned_count >= MAX_LIST_SCAN_ENTRIES:
                raise WorkspaceRepositoryError("workspace listing exceeds the safe scan bound")
            seen_cursors.add(next_cursor)
            repository_cursor = next_cursor
        matched.sort(key=lambda entry: (entry.path.casefold(), entry.entry_id))
        start = 0
        if cursor is not None:
            positions = [
                index for index, entry in enumerate(matched) if entry.entry_id == cursor
            ]
            if not positions:
                raise RepositoryScopeError("cursor does not belong to the bound listing")
            start = positions[0] + 1
        selected = tuple(matched[start : start + page_limit])
        has_more = start + len(selected) < len(matched)
        return MemoryEntryPage(
            entries=selected,
            next_cursor=selected[-1].entry_id if selected and has_more else None,
            has_more=has_more,
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        tags: Sequence[str] = (),
        backlink_ref: ResourceRef | None = None,
        recent_first: bool = False,
    ) -> WorkspaceSearchResult:
        return self._search.search(
            query,
            limit=limit,
            tags=tags,
            backlink_ref=backlink_ref,
            recent_first=recent_first,
        )

    def read(
        self,
        ref: ResourceRef,
        *,
        offset: int = 0,
        limit: int = MAX_READ_BYTES,
    ) -> WorkspaceContentPage:
        entry = self._read_entry_ref(ref)
        page_offset = _bounded_offset(offset)
        page_limit = _bounded_limit(limit, maximum=MAX_READ_BYTES)
        if entry.kind is MemoryEntryKind.FOLDER:
            raise RepositoryNotFoundError("folder entries do not have readable content")
        if entry.kind is MemoryEntryKind.LINK:
            payload = entry.link_url.encode("utf-8")
            return WorkspaceContentPage(
                ref=ref,
                media_type="text/uri-list",
                data=payload[page_offset : page_offset + page_limit],
                offset=min(page_offset, len(payload)),
                total_bytes=len(payload),
            )
        if entry.content_ref is None:
            raise WorkspaceRepositoryError("entry content reference is unavailable")
        page = self._content.read_content(
            ref=entry.content_ref,
            offset=page_offset,
            limit=page_limit,
        )
        if (
            page.ref != entry.content_ref
            or page.offset != page_offset
            or len(page.data) > page_limit
        ):
            raise WorkspaceRepositoryError("content capability returned an invalid page")
        media_type = entry.media_type or page.media_type
        return page if page.media_type == media_type else replace(page, media_type=media_type)

    def history(
        self,
        ref: ResourceRef,
        *,
        limit: int = 20,
        before_revision: int | None = None,
    ) -> tuple[MemoryEntry, ...]:
        current = self._read_entry_ref(ref)
        page_limit = _bounded_limit(limit, maximum=MAX_HISTORY_REVISIONS)
        if before_revision is not None:
            before_revision = _positive_revision(before_revision, "before_revision")
        revisions = self._history.list_revisions(
            ref=ref,
            before_revision=before_revision,
            limit=page_limit,
        )
        previous = (before_revision if before_revision is not None else current.revision + 1)
        for revision in revisions:
            self._validate_entry(revision)
            if revision.entry_id != current.entry_id or revision.revision >= previous:
                raise WorkspaceRepositoryError("history capability returned an invalid revision")
            previous = revision.revision
        if len(revisions) > page_limit:
            raise WorkspaceRepositoryError("history capability exceeded the requested limit")
        return revisions

    def _read_entry_ref(self, ref: ResourceRef) -> MemoryEntry:
        self._require_bound_entry_ref(ref)
        entry = self._repository.read_entry(ref=ref)
        self._validate_entry(entry)
        if entry.entry_id != ref.resource_id or entry.revision != ref.revision:
            raise RepositoryScopeError("repository returned a different entry revision")
        return entry

    def _require_bound_entry_ref(self, ref: ResourceRef) -> None:
        if not isinstance(ref, ResourceRef):
            raise TypeError("ref must be a ResourceRef")
        if ref.kind != "memory" or ref.fragment != self.space.space_id:
            raise RepositoryScopeError("reference does not belong to the bound workspace")

    def _validate_entry(self, entry: MemoryEntry) -> None:
        if not isinstance(entry, MemoryEntry):
            raise TypeError("repository returned a non-entry result")
        if entry.space_id != self.space.space_id:
            raise RepositoryScopeError("repository returned a foreign workspace entry")
        canonical_entry_path(entry.path)
        if entry.name != virtual_name(entry.path):
            raise WorkspaceRepositoryError("entry name does not match its virtual path")

    def _require_same_space(self, capability: object, label: str) -> None:
        capability_space = getattr(capability, "space", None)
        if not isinstance(capability_space, MemorySpace):
            raise TypeError(f"{label} must expose a bound MemorySpace")
        if capability_space.space_id != self.space.space_id:
            raise RepositoryScopeError(f"{label} belongs to another workspace")


class MemoryWorkspaceService(_BoundWorkspaceReads):
    """Provider-neutral chat workspace behavior over scope-bound capabilities."""

    def __init__(
        self,
        *,
        repository: BoundMemoryWorkspaceRepository,
        mutations: BoundWorkspaceMutationRepository,
        content: BoundWorkspaceContentRepository,
        history: BoundWorkspaceHistoryRepository,
        links: BoundWorkspaceLinkRepository,
        references: BoundWorkspaceReferenceAuthorizer,
        search: WorkspaceSearchService | None = None,
        content_redactor: Callable[[WorkspaceWriteDraft], WorkspaceWriteDraft]
        | None = None,
    ) -> None:
        super().__init__(
            repository=repository,
            content=content,
            history=history,
            search=search,
        )
        if not isinstance(mutations, BoundWorkspaceMutationRepository):
            raise TypeError("mutations must be a BoundWorkspaceMutationRepository")
        if not isinstance(references, BoundWorkspaceReferenceAuthorizer):
            raise TypeError("references must be a BoundWorkspaceReferenceAuthorizer")
        self._require_same_space(mutations, "mutations")
        if not isinstance(links, BoundWorkspaceLinkRepository):
            raise TypeError("links must be a BoundWorkspaceLinkRepository")
        self._require_same_space(links, "links")
        # Extend the validated search service with link filtering; retain its
        # existing repository, index providers, and search configuration.
        self._search = self._search.with_link_repository(links)
        self._mutations = mutations
        self._links = links
        self._references = references
        if content_redactor is not None and not callable(content_redactor):
            raise TypeError("content_redactor must be callable")
        self._content_redactor = content_redactor or _identity_content_redactor

    @property
    def binding_id(self) -> str:
        return self._references.binding_id

    def create_folder(
        self,
        *,
        path: str,
        description: str,
        expected_space_revision: int,
        source_refs: Sequence[ResourceRef],
        operation_id: str,
        tags: Sequence[str] | None = None,
    ) -> MemoryEntry:
        return self._write(
            entry_ref=None,
            path=path,
            description=description,
            kind=MemoryEntryKind.FOLDER,
            expected_space_revision=expected_space_revision,
            source_refs=source_refs,
            operation_id=operation_id,
            tags=tags,
        )

    def write_markdown(
        self,
        *,
        path: str,
        description: str,
        content: str | bytes,
        expected_space_revision: int,
        source_refs: Sequence[ResourceRef],
        operation_id: str,
        entry_ref: ResourceRef | None = None,
        tags: Sequence[str] | None = None,
    ) -> MemoryEntry:
        if isinstance(content, str):
            payload = content.encode("utf-8")
        elif isinstance(content, bytes):
            payload = content
        else:
            raise TypeError("content must be text or bytes")
        self._validate_write_content(payload)
        return self._write(
            entry_ref=entry_ref,
            path=path,
            description=description,
            kind=MemoryEntryKind.MARKDOWN,
            expected_space_revision=expected_space_revision,
            source_refs=source_refs,
            operation_id=operation_id,
            content=payload,
            media_type="text/markdown",
            tags=tags,
        )

    def write_image(
        self,
        *,
        path: str,
        description: str,
        content: bytes,
        media_type: str,
        expected_space_revision: int,
        source_refs: Sequence[ResourceRef],
        operation_id: str,
        entry_ref: ResourceRef | None = None,
        tags: Sequence[str] | None = None,
    ) -> MemoryEntry:
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        self._validate_write_content(content)
        normalized_media_type = _required_text(
            media_type,
            "media_type",
            maximum=255,
        ).casefold()
        if not normalized_media_type.startswith("image/"):
            raise ModelValidationError("image entries require an image MIME type")
        return self._write(
            entry_ref=entry_ref,
            path=path,
            description=description,
            kind=MemoryEntryKind.IMAGE,
            expected_space_revision=expected_space_revision,
            source_refs=source_refs,
            operation_id=operation_id,
            content=content,
            media_type=normalized_media_type,
            tags=tags,
        )

    def create_link(
        self,
        *,
        path: str,
        description: str,
        url: str,
        expected_space_revision: int,
        source_refs: Sequence[ResourceRef],
        operation_id: str,
        entry_ref: ResourceRef | None = None,
        tags: Sequence[str] | None = None,
    ) -> MemoryEntry:
        return self._write(
            entry_ref=entry_ref,
            path=path,
            description=description,
            kind=MemoryEntryKind.LINK,
            expected_space_revision=expected_space_revision,
            source_refs=source_refs,
            operation_id=operation_id,
            link_url=canonical_memory_link_url(url),
            tags=tags,
        )

    def move(
        self,
        *,
        ref: ResourceRef,
        new_path: str,
        expected_space_revision: int,
        source_refs: Sequence[ResourceRef],
        operation_id: str,
    ) -> MemoryEntry:
        current = self._read_entry_ref(ref)
        if current.deleted:
            raise RepositoryConflictError("archived entries cannot be moved")
        path = canonical_entry_path(new_path)
        expected_space = _positive_revision(
            expected_space_revision,
            "expected_space_revision",
        )
        provenance = self._authorize_source_refs(source_refs)
        if not provenance:
            raise ModelValidationError("entry revisions require fresh source provenance")
        operation = build_operation_ref(
            operation_id,
            domain="workspace.entry.move",
            payload={
                "ref": ref,
                "new_path": path,
                "expected_space_revision": expected_space,
                "source_refs": provenance,
            },
        )
        next_entry = replace(
            current,
            path=path,
            name=virtual_name(path),
            revision=current.revision + 1,
            updated_seq=expected_space + 1,
            source_refs=provenance,
        )
        return self._apply(
            entry=next_entry,
            expected_revision=current.revision,
            expected_space_revision=expected_space,
            operation=operation,
        )

    def archive(
        self,
        *,
        ref: ResourceRef,
        expected_space_revision: int,
        source_refs: Sequence[ResourceRef],
        operation_id: str,
        recursive: bool = False,
    ) -> MemoryEntry:
        current = self._read_entry_ref(ref)
        if current.deleted:
            raise RepositoryConflictError("entry is already archived")
        if not isinstance(recursive, bool):
            raise TypeError("recursive must be a boolean")
        if recursive and current.kind is not MemoryEntryKind.FOLDER:
            raise RepositoryConflictError("recursive archive requires a folder entry")
        expected_space = _positive_revision(
            expected_space_revision,
            "expected_space_revision",
        )
        provenance = self._authorize_source_refs(source_refs)
        if not provenance:
            raise ModelValidationError("entry revisions require fresh source provenance")
        operation = build_operation_ref(
            operation_id,
            domain="workspace.entry.archive",
            payload={
                "ref": ref,
                "recursive": recursive,
                "expected_space_revision": expected_space,
                "source_refs": provenance,
            },
        )
        return self._apply(
            entry=replace(
                current,
                revision=current.revision + 1,
                updated_seq=expected_space + 1,
                source_refs=provenance,
                deleted=True,
            ),
            expected_revision=current.revision,
            expected_space_revision=expected_space,
            operation=operation,
            recursive=recursive,
        )

    def link(
        self,
        *,
        source_ref: ResourceRef,
        target_ref: ResourceRef,
        relation: str,
        expected_space_revision: int,
        source_refs: Sequence[ResourceRef],
        operation_id: str,
    ) -> MemoryLink:
        self._read_current_entry_ref(source_ref)
        self._read_current_entry_ref(target_ref)
        normalized_relation = _required_text(
            relation,
            "relation",
            maximum=128,
            identifier=True,
        )
        provenance = self._authorize_source_refs(source_refs)
        if not provenance:
            raise ModelValidationError("relation links require source provenance")
        expected_space = _positive_revision(
            expected_space_revision,
            "expected_space_revision",
        )
        operation = build_operation_ref(
            operation_id,
            domain="workspace.relation.create",
            payload={
                "source_ref": source_ref,
                "target_ref": target_ref,
                "relation": normalized_relation,
                "expected_space_revision": expected_space,
                "source_refs": provenance,
            },
        )
        link = MemoryLink(
            link_id=_stable_id("link", self.space.space_id, operation),
            source_entry_ref=source_ref,
            target_ref=target_ref,
            relation=normalized_relation,
            revision=1,
        )
        persisted = self._links.create_link(
            request=WorkspaceLinkRequest(
                link=link,
                source_entry_ref=source_ref,
                expected_space_revision=expected_space,
                source_refs=provenance,
                operation=operation,
            )
        )
        if persisted != link:
            raise WorkspaceRepositoryError("link capability returned a divergent relation")
        return persisted

    def list_links(
        self,
        source_ref: ResourceRef,
        *,
        limit: int = 100,
    ) -> tuple[MemoryLink, ...]:
        self._read_current_entry_ref(source_ref)
        page_limit = _bounded_limit(limit, maximum=MAX_LIST_RESULTS)
        links = self._links.list_links(
            source_entry_ref=source_ref,
            limit=page_limit,
        )
        if len(links) > page_limit:
            raise WorkspaceRepositoryError("link capability exceeded the requested limit")
        for link in links:
            if not isinstance(link, MemoryLink) or link.source_entry_ref != source_ref:
                raise RepositoryScopeError("link capability returned a foreign relation")
            self._require_bound_entry_ref(link.target_ref)
            self._read_entry_ref(link.target_ref)
        return links

    def _read_current_entry_ref(self, ref: ResourceRef) -> MemoryEntry:
        historical = self._read_entry_ref(ref)
        current = self._repository.read_current_entry(entry_id=ref.resource_id)
        self._validate_entry(current)
        if current.entry_id != ref.resource_id:
            raise RepositoryScopeError("repository returned a different current entry")
        if current.deleted:
            raise RepositoryConflictError("archived entries cannot be used")
        if current.revision != ref.revision or current != historical:
            raise RepositoryConflictError("entry reference is not current")
        return current

    def _write(
        self,
        *,
        entry_ref: ResourceRef | None,
        path: str,
        description: str,
        kind: MemoryEntryKind,
        expected_space_revision: int,
        source_refs: Sequence[ResourceRef],
        operation_id: str,
        content: bytes | None = None,
        media_type: str = "",
        link_url: str = "",
        tags: Sequence[str] | None = None,
    ) -> MemoryEntry:
        canonical_path = canonical_entry_path(path)
        canonical_description = meaningful_description(description)
        expected_space = _positive_revision(
            expected_space_revision,
            "expected_space_revision",
        )
        authorized_sources = self._authorize_source_refs(source_refs)
        if not authorized_sources:
            raise ModelValidationError("entry revisions require fresh source provenance")
        current: MemoryEntry | None = None
        if entry_ref is not None:
            current = self._read_entry_ref(entry_ref)
            if current.deleted:
                raise RepositoryConflictError("archived entries cannot be updated")
            if current.kind is not kind:
                raise RepositoryConflictError("entry kind cannot change")
        normalized_media_type = (
            _required_text(media_type, "media_type", maximum=255).casefold()
            if media_type
            else ""
        )
        normalized_link_url = (
            canonical_memory_link_url(link_url) if link_url else ""
        )
        normalized_tags = (
            canonical_memory_tags(tags)
            if tags is not None
            else (current.tags if current is not None else ())
        )
        draft = WorkspaceWriteDraft(
            path=canonical_path,
            description=canonical_description,
            kind=kind,
            content=content,
            media_type=normalized_media_type,
            link_url=normalized_link_url,
            tags=normalized_tags,
        )
        try:
            redacted = self._content_redactor(draft)
        except Exception:
            raise WorkspaceRepositoryError(
                "workspace content redaction failed"
            ) from None
        if (
            not isinstance(redacted, WorkspaceWriteDraft)
            or redacted.kind is not kind
            or (redacted.content is None) != (draft.content is None)
        ):
            raise WorkspaceRepositoryError("workspace content redaction failed")
        try:
            canonical_path = canonical_entry_path(redacted.path)
            canonical_description = meaningful_description(redacted.description)
            normalized_media_type = (
                _required_text(
                    redacted.media_type,
                    "media_type",
                    maximum=255,
                ).casefold()
                if redacted.media_type
                else ""
            )
            normalized_link_url = (
                canonical_memory_link_url(redacted.link_url)
                if redacted.link_url
                else ""
            )
            normalized_tags = canonical_memory_tags(redacted.tags)
            content = redacted.content
            if content is not None:
                self._validate_write_content(content)
            if kind is MemoryEntryKind.IMAGE and not normalized_media_type.startswith(
                "image/"
            ):
                raise ModelValidationError(
                    "image entries require an image MIME type"
                )
        except Exception:
            raise WorkspaceRepositoryError(
                "workspace content redaction failed"
            ) from None
        operation = build_operation_ref(
            operation_id,
            domain="workspace.entry.write",
            payload={
                "entry_ref": entry_ref,
                "path": canonical_path,
                "description": canonical_description,
                "kind": kind,
                "expected_space_revision": expected_space,
                "source_refs": authorized_sources,
                "content_sha256": (
                    hashlib.sha256(content).hexdigest() if content is not None else ""
                ),
                "content_bytes": len(content) if content is not None else 0,
                "media_type": normalized_media_type,
                "link_url": normalized_link_url,
                "tags": normalized_tags,
            },
        )
        entry = MemoryEntry(
            entry_id=(
                current.entry_id
                if current is not None
                else _stable_id("memory", self.space.space_id, operation)
            ),
            space_id=self.space.space_id,
            path=canonical_path,
            name=virtual_name(canonical_path),
            description=canonical_description,
            kind=kind,
            revision=current.revision + 1 if current is not None else 1,
            updated_seq=expected_space + 1,
            content_ref=current.content_ref if current is not None else None,
            source_refs=authorized_sources,
            tags=normalized_tags,
            media_type=normalized_media_type,
            link_url=normalized_link_url,
        )
        return self._apply(
            entry=entry,
            expected_revision=current.revision if current is not None else None,
            expected_space_revision=expected_space,
            operation=operation,
            content=content,
        )

    def _apply(
        self,
        *,
        entry: MemoryEntry,
        expected_revision: int | None,
        expected_space_revision: int,
        operation: OperationRef,
        content: bytes | None = None,
        recursive: bool = False,
    ) -> MemoryEntry:
        expected_space = _positive_revision(
            expected_space_revision,
            "expected_space_revision",
        )
        expected_updated_seq = expected_space + 1
        if entry.updated_seq != expected_updated_seq:
            raise WorkspaceRepositoryError(
                "mutation entry has a divergent updated sequence"
            )
        persisted = self._mutations.apply(
            request=WorkspaceMutationRequest(
                entry=entry,
                expected_revision=expected_revision,
                expected_space_revision=expected_space,
                operation=operation,
                content=content,
                recursive=recursive,
            )
        )
        self._validate_entry(persisted)
        if (
            persisted.entry_id != entry.entry_id
            or persisted.revision != entry.revision
            or persisted.updated_seq != expected_updated_seq
            or persisted.path != entry.path
            or persisted.name != entry.name
            or persisted.description != entry.description
            or persisted.kind is not entry.kind
            or persisted.tags != entry.tags
            or persisted.media_type != entry.media_type
            or persisted.link_url != entry.link_url
            or persisted.deleted != entry.deleted
            or persisted.source_refs != entry.source_refs
            or (content is None and persisted.content_ref != entry.content_ref)
        ):
            raise WorkspaceRepositoryError("mutation capability returned a divergent entry")
        if entry.kind in {MemoryEntryKind.MARKDOWN, MemoryEntryKind.IMAGE} and (
            persisted.content_ref is None
        ):
            raise WorkspaceRepositoryError("content mutation did not return a durable reference")
        self._search.index_entry(persisted, content=content)
        return persisted

    def _authorize_source_refs(
        self,
        source_refs: Sequence[ResourceRef],
    ) -> tuple[ResourceRef, ...]:
        if isinstance(source_refs, (str, bytes, bytearray)):
            raise TypeError("source_refs must be a sequence of ResourceRef values")
        authorized: list[ResourceRef] = []
        for raw_ref in source_refs:
            ref = raw_ref if isinstance(raw_ref, ResourceRef) else ResourceRef.from_dict(raw_ref)
            if ref.kind not in _SOURCE_KINDS:
                raise RepositoryScopeError("source reference kind is not allowed")
            canonical = self._references.authorize(ref=ref)
            if canonical != ref:
                raise RepositoryScopeError("reference authorizer changed the source reference")
            authorized.append(ref)
        return tuple(authorized)

    @staticmethod
    def _merge_refs(
        first: Sequence[ResourceRef],
        second: Sequence[ResourceRef],
    ) -> tuple[ResourceRef, ...]:
        merged: list[ResourceRef] = []
        for ref in (*first, *second):
            if ref not in merged:
                merged.append(ref)
        return tuple(merged)

    @staticmethod
    def _validate_write_content(content: bytes) -> None:
        if not content:
            raise ModelValidationError("content must not be empty")
        if len(content) > MAX_WRITE_BYTES:
            raise ModelValidationError(
                f"content must not exceed {MAX_WRITE_BYTES} bytes"
            )


class LongTermMemoryService(_BoundWorkspaceReads):
    """Read-only capability for one host-bound long-term namespace."""

    def __init__(
        self,
        *,
        binding_id: str,
        repository: BoundMemoryWorkspaceRepository,
        content: BoundWorkspaceContentRepository,
        history: BoundWorkspaceHistoryRepository,
        search: WorkspaceSearchService | None = None,
    ) -> None:
        self._binding_id = _required_text(
            binding_id,
            "binding_id",
            maximum=512,
            identifier=True,
        )
        super().__init__(
            repository=repository,
            content=content,
            history=history,
            search=search,
        )
        if self.space.namespace == "chat":
            raise RepositoryScopeError("long-term capability requires a namespaced space")

    @property
    def namespace(self) -> str:
        return self.space.namespace

    @property
    def binding_id(self) -> str:
        return self._binding_id


__all__ = [
    "LongTermMemoryService",
    "MAX_HISTORY_REVISIONS",
    "MAX_LIST_RESULTS",
    "MAX_LIST_SCAN_ENTRIES",
    "MAX_READ_BYTES",
    "MAX_WRITE_BYTES",
    "MemoryWorkspaceService",
    "WorkspaceWriteDraft",
]
