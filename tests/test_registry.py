"""Tests for the Registry pattern implementation."""

from __future__ import annotations

import pytest

from biometric.data.registry import Registry


class TestRegistry:
    """Tests for the generic Registry class."""

    def test_register_and_get(self) -> None:
        registry: Registry = Registry("TestRegistry")

        @registry.register("my_class")
        class MyClass:
            pass

        retrieved = registry.get("my_class")
        assert retrieved is MyClass

    def test_duplicate_registration_raises(self) -> None:
        registry: Registry = Registry("TestRegistry")

        @registry.register("dup")
        class First:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @registry.register("dup")
            class Second:
                pass

    def test_get_missing_key_raises(self) -> None:
        registry: Registry = Registry("TestRegistry")
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_registered(self) -> None:
        registry: Registry = Registry("TestRegistry")

        @registry.register("b_class")
        class B:
            pass

        @registry.register("a_class")
        class A:
            pass

        assert registry.list_registered() == ["a_class", "b_class"]

    def test_contains(self) -> None:
        registry: Registry = Registry("TestRegistry")

        @registry.register("exists")
        class Exists:
            pass

        assert "exists" in registry
        assert "missing" not in registry

    def test_repr(self) -> None:
        registry: Registry = Registry("MyReg")

        @registry.register("item")
        class Item:
            pass

        repr_str = repr(registry)
        assert "MyReg" in repr_str
        assert "item" in repr_str

    def test_error_message_shows_available(self) -> None:
        registry: Registry = Registry("TestRegistry")

        @registry.register("available_one")
        class One:
            pass

        @registry.register("available_two")
        class Two:
            pass

        with pytest.raises(KeyError, match="available_one"):
            registry.get("missing")
