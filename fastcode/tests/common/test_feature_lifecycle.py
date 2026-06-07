"""Tests for the capability lifecycle registry."""

from __future__ import annotations

import warnings

import pytest

from fastcode.common.feature_lifecycle import (
    CapabilityLookupError,
    CapabilityRemovedError,
    CapabilitySpec,
    CapabilityStage,
    _Registry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> _Registry:
    return _Registry()


def _stable(name: str = "cap_a", **kw: object) -> CapabilitySpec:
    return CapabilitySpec(
        stage=CapabilityStage.STABLE, name=name, description="test", **kw
    )


def _experimental(name: str = "cap_exp", **kw: object) -> CapabilitySpec:
    return CapabilitySpec(
        stage=CapabilityStage.EXPERIMENTAL, name=name, description="test", **kw
    )


def _deprecated(name: str = "cap_dep", **kw: object) -> CapabilitySpec:
    return CapabilitySpec(
        stage=CapabilityStage.DEPRECATED,
        name=name,
        description="test",
        deprecated_in="0.5.0",
        replacement="cap_new",
        **kw,
    )


def _removed(name: str = "cap_old", **kw: object) -> CapabilitySpec:
    return CapabilitySpec(
        stage=CapabilityStage.REMOVED,
        name=name,
        description="test",
        removed_in="1.0.0",
        replacement="cap_new",
        **kw,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_and_get(self, registry: _Registry) -> None:
        spec = _stable()
        registry.register(spec)
        assert registry.get("cap_a") is spec

    def test_duplicate_name_raises(self, registry: _Registry) -> None:
        registry.register(_stable())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_stable())

    def test_get_unknown_raises(self, registry: _Registry) -> None:
        with pytest.raises(CapabilityLookupError):
            registry.get("nonexistent")


# ---------------------------------------------------------------------------
# check() behaviour per stage
# ---------------------------------------------------------------------------


class TestCheckStable:
    def test_stable_is_silent(self, registry: _Registry) -> None:
        registry.register(_stable())
        registry.check("cap_a")  # no warning, no error


class TestCheckExperimental:
    def test_warns_once(self, registry: _Registry) -> None:
        registry.register(_experimental())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            registry.check("cap_exp")
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "experimental" in str(w[0].message)
        # second call: no additional warning
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            registry.check("cap_exp")
            assert len(w2) == 0


class TestCheckDeprecated:
    def test_emits_deprecation_warning(self, registry: _Registry) -> None:
        registry.register(_deprecated())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            registry.check("cap_dep")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            msg = str(w[0].message)
            assert "deprecated" in msg
            assert "0.5.0" in msg
            assert "cap_new" in msg

    def test_warns_on_every_call(self, registry: _Registry) -> None:
        registry.register(_deprecated())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            registry.check("cap_dep")
            registry.check("cap_dep")
            assert len(w) == 2


class TestCheckRemoved:
    def test_raises(self, registry: _Registry) -> None:
        registry.register(_removed())
        with pytest.raises(CapabilityRemovedError, match="removed"):
            registry.check("cap_old")

    def test_error_includes_replacement(self, registry: _Registry) -> None:
        registry.register(_removed())
        with pytest.raises(CapabilityRemovedError) as exc_info:
            registry.check("cap_old")
        assert "cap_new" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


class TestQueries:
    def test_all_by_stage(self, registry: _Registry) -> None:
        registry.register(_stable(name="a"))
        registry.register(_experimental(name="b"))
        registry.register(_deprecated(name="c"))
        assert len(registry.all_by_stage(CapabilityStage.STABLE)) == 1
        assert len(registry.all_by_stage(CapabilityStage.EXPERIMENTAL)) == 1
        assert len(registry.all_by_stage(CapabilityStage.DEPRECATED)) == 1

    def test_all_capabilities(self, registry: _Registry) -> None:
        registry.register(_stable(name="a"))
        registry.register(_experimental(name="b"))
        assert len(registry.all_capabilities()) == 2

    def test_clear(self, registry: _Registry) -> None:
        registry.register(_stable())
        registry.clear()
        assert registry.all_capabilities() == []
        # experimental warning state also cleared
        registry.register(_experimental())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            registry.check("cap_exp")
            assert len(w) == 1


# ---------------------------------------------------------------------------
# CapabilitySpec frozen
# ---------------------------------------------------------------------------


class TestSpecFrozen:
    def test_frozen(self) -> None:
        spec = _stable()
        with pytest.raises(AttributeError):
            spec.name = "changed"  # type: ignore[misc]
