"""Controlled vocabulary for the scope (context) keys of extracted knowledge.

An ontology lists, per context key, the canonical values a knowledge base
knows, their aliases, and the other keys each value implies (a component's
machine, a machine's line). Ingestion normalizes extracted context to it, so
filters and duplicate detection see one spelling per asset. Values the
ontology does not know are kept and reported, for an expert to add or fix.

File format (YAML):

    keys:
      <context key>:
        # Closed keys only accept listed values; the extraction schema
        # offers them to the LLM as an enum.
        closed: false
        description: optional text shown to the LLM
        values:
          <canonical value>:
            aliases: [other spellings]
            description: optional text shown to the LLM
            implies: {<other context key>: <value>}
          <canonical value without details>:
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Spelling differences that never distinguish two values: case, spaces,
# hyphens, underscores, dots ("AX 220", "ax-220" and "AX220" are one model).
_FOLD_RE = re.compile(r"[\s\-_.]+")


def _fold(value: str) -> str:
    return _FOLD_RE.sub("", value.casefold())


@dataclass(frozen=True)
class OntologyKey:
    """Known values of one context key."""

    name: str
    closed: bool = False
    description: str = ""
    # Canonical value -> (other key -> implied value).
    implies: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    descriptions: Mapping[str, str] = field(default_factory=dict)
    # Folded spelling (canonical or alias) -> canonical value.
    index: Mapping[str, str] = field(default_factory=dict)

    @property
    def values(self) -> list[str]:
        return list(self.implies)

    def resolve(self, value: str) -> str | None:
        return self.index.get(_fold(value))


@dataclass(frozen=True)
class OntologyReport:
    """What normalization changed or could not resolve in one context."""

    # key -> the spelling the extraction used before it was made canonical.
    normalized: dict[str, str] = field(default_factory=dict)
    # key -> value filled in because another value implies it.
    implied: dict[str, str] = field(default_factory=dict)
    # key -> value the ontology does not list (kept as extracted).
    unknown: dict[str, str] = field(default_factory=dict)
    # key -> the key whose values include it, e.g. a sensor under machine_id.
    misplaced: dict[str, str] = field(default_factory=dict)
    # key -> {"stated": extracted value, "implied": value, "by": "key=value"}.
    conflicts: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Non-empty sections only; empty when nothing needs attention."""

        return {
            name: section
            for name, section in (
                ("normalized", self.normalized),
                ("implied", self.implied),
                ("unknown", self.unknown),
                ("misplaced", self.misplaced),
                ("conflicts", self.conflicts),
            )
            if section
        }

    @property
    def needs_review(self) -> bool:
        return bool(self.unknown or self.misplaced or self.conflicts)


class Ontology:
    """Normalizes context dictionaries to a knowledge base's known values."""

    def __init__(self, keys: Mapping[str, OntologyKey], *, source: str = "") -> None:
        self._keys = dict(keys)
        self._source = source
        for key in self._keys.values():
            for value, implied in key.implies.items():
                for implied_key, implied_value in implied.items():
                    if implied_key == key.name:
                        raise ValueError(f"{key.name}={value} implies its own key")
                    target = self._keys.get(implied_key)
                    if target is not None and target.resolve(implied_value) != implied_value:
                        raise ValueError(
                            f"{key.name}={value} implies {implied_key}={implied_value}, "
                            f"which is not a canonical {implied_key}"
                        )

    @classmethod
    def from_file(cls, path: Path) -> "Ontology":
        source = path.read_text(encoding="utf-8")
        return cls.from_dict(yaml.safe_load(source) or {}, source=source)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source: str = "") -> "Ontology":
        keys: dict[str, OntologyKey] = {}
        for name, spec in (data.get("keys") or {}).items():
            spec = spec or {}
            implies: dict[str, dict[str, str]] = {}
            descriptions: dict[str, str] = {}
            index: dict[str, str] = {}
            for value, details in (spec.get("values") or {}).items():
                value = str(value)
                details = details or {}
                implies[value] = {
                    str(k): str(v) for k, v in (details.get("implies") or {}).items()
                }
                if description := str(details.get("description") or "").strip():
                    descriptions[value] = description
                for spelling in (value, *map(str, details.get("aliases") or ())):
                    folded = _fold(spelling)
                    if index.setdefault(folded, value) != value:
                        raise ValueError(
                            f"{name}: {spelling!r} is a spelling of both "
                            f"{index[folded]!r} and {value!r}"
                        )
            keys[name] = OntologyKey(
                name=name,
                closed=bool(spec.get("closed", False)),
                description=str(spec.get("description") or "").strip(),
                implies=implies,
                descriptions=descriptions,
                index=index,
            )
        return cls(keys, source=source)

    @property
    def keys(self) -> Mapping[str, OntologyKey]:
        return self._keys

    @property
    def sha256(self) -> str:
        """Identifies the ontology version that normalized a record."""

        return hashlib.sha256(self._source.encode("utf-8")).hexdigest()

    def normalize(self, context: Mapping[str, Any]) -> tuple[dict[str, Any], OntologyReport]:
        """Canonical copy of `context` and a report of what changed or is unknown.

        Implied keys are only filled when the context leaves them out; a
        different stated value is kept and reported as a conflict.
        """

        result = dict(context)
        report = OntologyReport()
        resolved: list[tuple[str, str]] = []
        for name, key in self._keys.items():
            value = result.get(name)
            if not isinstance(value, str) or not value.strip():
                continue
            canonical = key.resolve(value)
            if canonical is None:
                owner = next(
                    (other.name for other in self._keys.values()
                     if other.name != name and other.resolve(value) is not None),
                    None,
                )
                if owner is not None:
                    report.misplaced[name] = owner
                else:
                    report.unknown[name] = value
                continue
            if canonical != value:
                report.normalized[name] = value
                result[name] = canonical
            resolved.append((name, canonical))

        # Breadth-first, so a component implies its machine, then the line.
        while resolved:
            name, value = resolved.pop(0)
            for implied_key, implied_value in self._keys[name].implies[value].items():
                stated = result.get(implied_key)
                if stated is None or (isinstance(stated, str) and not stated.strip()):
                    result[implied_key] = implied_value
                    report.implied[implied_key] = implied_value
                    if implied_key in self._keys:
                        resolved.append((implied_key, implied_value))
                elif stated != implied_value and implied_key not in report.conflicts:
                    report.conflicts[implied_key] = {
                        "stated": str(stated), "implied": implied_value, "by": f"{name}={value}",
                    }
        return result, report

    def constrain_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Copy of a JSON schema whose closed keys only accept listed values.

        Every object property named after a closed key that accepts a string
        gets the canonical values as an enum, described for the LLM.
        """

        schema = copy.deepcopy(schema)
        closed = {name: key for name, key in self._keys.items() if key.closed and key.values}

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                for name, prop in (node.get("properties") or {}).items():
                    if name in closed and isinstance(prop, dict):
                        _constrain(prop, closed[name])
                for child in node.values():
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        visit(schema)
        return schema


def _constrain(prop: dict[str, Any], key: OntologyKey) -> None:
    lines = [key.description] if key.description else []
    lines += [
        f"{value}: {key.descriptions[value]}" for value in key.values if value in key.descriptions
    ]
    constrained = {"type": "string", "enum": key.values}
    if lines:
        constrained["description"] = "\n".join(lines)
    options = prop.get("anyOf")
    if isinstance(options, list):
        prop["anyOf"] = [
            constrained if option.get("type") == "string" else option for option in options
        ]
    elif prop.get("type") == "string":
        prop.update(constrained)
