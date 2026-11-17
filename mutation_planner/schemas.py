"""Pydantic models shared by every stage of the planner.

Three groups of models live here:

* Repository facts collected deterministically (``MutationContext`` and its parts).
  This is what ``--dry-run`` writes to ``mutation-context.json``.
* The object the LLM must return (``LLMPlan``). Its JSON schema is generated from the
  model and handed to the provider; the LLM never defines its own schema.
* The final, validated and enriched artifact (``MutationPlan``).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTEXT_SCHEMA_VERSION = "1.0"
PLAN_SCHEMA_VERSION = "1.0"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Repository facts
# ---------------------------------------------------------------------------


class LineRange(_Strict):
    start: int = Field(ge=1)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> "LineRange":
        if self.end < self.start:
            raise ValueError(f"line range end ({self.end}) is before start ({self.start})")
        return self

    def overlaps(self, other: "LineRange") -> bool:
        return self.start <= other.end and other.start <= self.end

    def contains(self, line: int) -> bool:
        return self.start <= line <= self.end


class ChangeStatus(str, Enum):
    added = "added"
    modified = "modified"
    deleted = "deleted"
    renamed = "renamed"
    copied = "copied"
    type_changed = "type_changed"


class FileClass(str, Enum):
    production_cpp = "production_cpp"
    production_header = "production_header"
    test_source = "test_source"
    documentation = "documentation"
    build_config = "build_config"
    generated = "generated"
    third_party = "third_party"
    other = "other"


PRODUCTION_CLASSES = {FileClass.production_cpp, FileClass.production_header}


class ChangedFile(_Strict):
    path: str = Field(description="Path at the PR head (old path for deleted files)")
    old_path: str | None = None
    status: ChangeStatus
    binary: bool = False
    classification: FileClass
    classification_reason: str
    additions: int = 0
    deletions: int = 0
    head_blob_sha: str | None = Field(default=None, description="Blob SHA at head; None when deleted")
    head_line_count: int | None = None
    changed_lines: list[LineRange] = Field(
        default_factory=list,
        description="Line ranges at head that were added or modified",
    )
    deletion_anchors: list[int] = Field(
        default_factory=list,
        description="Head lines after which lines were removed without replacement",
    )
    patch_excerpt: str | None = Field(
        default=None, description="Bounded diff excerpt for non-production files (production patches live in SourceFileContext)"
    )

    @property
    def exists_at_head(self) -> bool:
        return self.status != ChangeStatus.deleted and self.head_blob_sha is not None

    @property
    def is_production(self) -> bool:
        return self.classification in PRODUCTION_CLASSES


class ChangedFunction(_Strict):
    name: str = Field(description="Name as written at the definition, e.g. CWallet::AddToWallet")
    short_name: str
    enclosing_scope: list[str] = Field(default_factory=list, description="Namespaces/classes")
    kind: Literal["function", "method", "constructor", "destructor", "operator", "fuzz_target", "other"]
    start_line: int
    end_line: int
    changed_lines: list[LineRange]
    rpc_name: str | None = Field(default=None, description="RPC method defined by this function")
    snippet: str | None = None
    snippet_start_line: int | None = None
    snippet_truncated: bool = False


class SourceFileContext(_Strict):
    path: str
    classification: FileClass
    parse_ok: bool = Field(description="False when syntax-aware parsing failed or was unavailable")
    parse_notes: list[str] = Field(default_factory=list)
    functions: list[ChangedFunction] = Field(default_factory=list)
    unattributed_changed_lines: list[LineRange] = Field(
        default_factory=list, description="Changed lines outside any recognised function"
    )
    unattributed_snippets: list[str] = Field(default_factory=list)
    affected_classes: list[str] = Field(default_factory=list)
    changed_string_literals: list[str] = Field(default_factory=list)
    patch: str = ""
    patch_truncated: bool = False


class TestCategory(str, Enum):
    __test__ = False  # not a pytest test class

    unit = "unit"
    functional = "functional"
    fuzz = "fuzz"


class TestEntry(_Strict):
    __test__ = False

    id: str
    category: TestCategory
    source_path: str
    line: int | None = None
    end_line: int | None = None
    # unit
    suite: str | None = None
    case: str | None = None
    granularity: Literal["suite", "case", "script", "target"]
    # execution metadata (separate from the identifier)
    executable: str | None = None
    selector: str | None = Field(default=None, description="Runner-specific filter, e.g. --run_test=a/b")
    command: str | None = Field(default=None, description="Command template, relative to a build dir")
    build_condition: str | None = Field(default=None, description="CMake option the test depends on")
    registered_in_build: bool | None = None
    # functional
    extended: bool | None = None
    variants: list[str] = Field(default_factory=list)
    scenarios: list[str] = Field(default_factory=list)
    # fuzz
    hidden: bool | None = None
    init_function: str | None = None


class TestInventory(_Strict):
    __test__ = False

    unit: list[TestEntry] = Field(default_factory=list)
    functional: list[TestEntry] = Field(default_factory=list)
    fuzz: list[TestEntry] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def all(self) -> list[TestEntry]:
        return [*self.unit, *self.functional, *self.fuzz]

    def by_id(self) -> dict[str, TestEntry]:
        return {t.id: t for t in self.all()}


class Relationship(str, Enum):
    direct_symbol_reference = "direct_symbol_reference"
    class_reference = "class_reference"
    rpc_method_call = "rpc_method_call"
    string_literal_match = "string_literal_match"
    header_include = "header_include"
    call_path = "call_path"
    rpc_call_path = "rpc_call_path"
    module_area = "module_area"
    name_similarity = "name_similarity"
    coverage = "coverage"  # reserved for future dynamic coverage data


# Relationships that are verified static facts (the test text really references the code).
DIRECT_RELATIONSHIPS = {
    Relationship.direct_symbol_reference,
    Relationship.class_reference,
    Relationship.rpc_method_call,
    Relationship.string_literal_match,
    Relationship.header_include,
}
INFERRED_RELATIONSHIPS = {Relationship.call_path, Relationship.rpc_call_path, Relationship.module_area}
WEAK_RELATIONSHIPS = {Relationship.name_similarity}


class Evidence(_Strict):
    id: str
    source_file: str
    test_id: str
    relationship: Relationship
    basis: Literal["static", "inferred", "heuristic", "coverage"]
    path: str
    line: int | None = None
    symbol: str | None = None
    description: str


class Candidate(_Strict):
    source_file: str
    test_id: str
    category: TestCategory
    relationships: list[Relationship]
    strength: Literal["direct", "inferred", "weak"]
    score: float
    evidence: list[str]


class RepoSnapshot(_Strict):
    repo_name: str | None = None
    base_sha: str
    head_sha: str
    merge_base_sha: str
    head_tree_sha: str
    base_ref_input: str
    head_ref_input: str


class PRMetadata(_Strict):
    number: int | None = None
    title: str | None = None
    description: str | None = None


class TruncationRecord(_Strict):
    scope: str
    detail: str


class MutationContext(_Strict):
    schema_version: str = CONTEXT_SCHEMA_VERSION
    tool_version: str
    snapshot: RepoSnapshot
    pr: PRMetadata
    changed_files: list[ChangedFile]
    production_files: list[SourceFileContext]
    test_inventory: TestInventory
    evidence: list[Evidence]
    candidates: list[Candidate]
    limits: dict[str, Any]
    truncations: list[TruncationRecord] = Field(default_factory=list)
    analysis_complete: bool = True
    analysis_limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM response
# ---------------------------------------------------------------------------

Priority = Literal["high", "medium", "low"]
Confidence = Literal["high", "medium", "low"]


class LLMTestSelection(_Strict):
    id: str
    reason: str
    confidence: Confidence
    evidence: list[str] = Field(default_factory=list)


class LLMTests(_Strict):
    unit: list[LLMTestSelection] = Field(default_factory=list)
    functional: list[LLMTestSelection] = Field(default_factory=list)
    fuzz: list[LLMTestSelection] = Field(default_factory=list)


class LLMFileSelection(_Strict):
    path: str
    priority: Priority
    reason: str
    functions: list[str] = Field(default_factory=list)
    changed_lines: list[LineRange] = Field(default_factory=list)
    tests: LLMTests
    uncertainty: str | None = None


class Decision(_Strict):
    mutate: bool
    reason: str


class LLMPlan(_Strict):
    decision: Decision
    files: list[LLMFileSelection] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def llm_output_schema() -> dict[str, Any]:
    """JSON schema handed to providers. Inlines $defs for maximum provider compatibility."""
    schema = LLMPlan.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                return resolve(defs[name])
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


# ---------------------------------------------------------------------------
# Final artifact
# ---------------------------------------------------------------------------


class PlanTestSelection(LLMTestSelection):
    command: str | None = Field(default=None, description="Populated from the inventory, not the LLM")


class PlanTests(_Strict):
    unit: list[PlanTestSelection] = Field(default_factory=list)
    functional: list[PlanTestSelection] = Field(default_factory=list)
    fuzz: list[PlanTestSelection] = Field(default_factory=list)


class PlanFile(_Strict):
    path: str
    priority: Priority
    reason: str
    functions: list[str]
    changed_lines: list[LineRange]
    tests: PlanTests
    uncertainty: str | None = None


class PlanPR(_Strict):
    number: int | None
    base_sha: str
    head_sha: str
    merge_base_sha: str


class PlanMetadata(_Strict):
    provider: str
    model: str | None
    prompt_version: str
    prompt_sha256: str
    context_sha256: str
    tool_version: str
    head_tree_sha: str
    validation_status: Literal["valid"]
    llm_attempts: int


class MutationPlan(_Strict):
    schema_version: str = PLAN_SCHEMA_VERSION
    pr: PlanPR
    decision: Decision
    files: list[PlanFile]
    warnings: list[str]
    metadata: PlanMetadata
