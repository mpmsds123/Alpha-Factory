"""Stable enum values persisted by the Milestone 1 engine."""

from enum import StrEnum


class DataType(StrEnum):
    MATRIX = "MATRIX"
    VECTOR = "VECTOR"


class AdapterMode(StrEnum):
    REPLAY = "replay"
    LIVE = "live"


class RunStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CandidateStatus(StrEnum):
    PROPOSED = "PROPOSED"
    VALIDATING = "VALIDATING"
    READY = "READY"
    QUEUED = "QUEUED"
    SIMULATING = "SIMULATING"
    COMPLETED = "COMPLETED"
    TRIAGED = "TRIAGED"
    INVALID = "INVALID"
    DUPLICATE = "DUPLICATE"
    RETRYABLE_ERROR = "RETRYABLE_ERROR"
    TERMINAL_ERROR = "TERMINAL_ERROR"
    ARCHIVED = "ARCHIVED"
    READY_FOR_PLATFORM_CHECK = "READY_FOR_PLATFORM_CHECK"
    READY_FOR_HUMAN_REVIEW = "READY_FOR_HUMAN_REVIEW"


class AttemptStatus(StrEnum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    RETRYABLE_ERROR = "RETRYABLE_ERROR"
    TERMINAL_ERROR = "TERMINAL_ERROR"
    CANCELLED = "CANCELLED"


class ErrorClass(StrEnum):
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    HUMAN_VERIFICATION_REQUIRED = "HUMAN_VERIFICATION_REQUIRED"
    RATE_LIMIT = "RATE_LIMIT"
    TIMEOUT = "TIMEOUT"
    NETWORK = "NETWORK"
    PENDING = "PENDING"
    CANCELLED = "CANCELLED"
    INVALID_EXPRESSION = "INVALID_EXPRESSION"
    INVALID_FIELD = "INVALID_FIELD"
    INVALID_OPERATOR = "INVALID_OPERATOR"
    INVALID_SETTING = "INVALID_SETTING"
    PLATFORM_TERMINAL_FAILURE = "PLATFORM_TERMINAL_FAILURE"
    UNKNOWN = "UNKNOWN"


class GateOutcome(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    PENDING = "PENDING"
    ERROR = "ERROR"
    STRONG = "STRONG"
    WEAK = "WEAK"
    DISCARD = "DISCARD"
    REUSED = "REUSED"


class PlatformCheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    PENDING = "PENDING"
    ERROR = "ERROR"


class MethodCardStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    RETIRED = "retired"


class EvidenceLevel(StrEnum):
    OFFICIAL = "official"
    TRAINING = "training"
    FORUM_EXPERIENCE = "forum_experience"
    UNVERIFIED = "unverified"


class LLMRole(StrEnum):
    GENERATOR = "generator"
    REPAIR_PLANNER = "repair_planner"


class LLMValidationStatus(StrEnum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


class RepairAction(StrEnum):
    STOP = "STOP"
    REPAIR = "REPAIR"


class MetadataSnapshotKind(StrEnum):
    FIELDS = "FIELDS"
    OPERATORS = "OPERATORS"
    SETTINGS = "SETTINGS"
