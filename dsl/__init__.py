"""DSL çekirdeği — operatörler, compiler, static validator (deterministik, LLM'e kapalı)."""
from dsl.compiler import CompileError, compile_hypothesis
from dsl.static_validator import validate

__all__ = ["compile_hypothesis", "CompileError", "validate"]
