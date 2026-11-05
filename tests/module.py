"""Core module dispatch, composition, and isolated scoped configuration."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from dataclasses import dataclass
from inspect import isabstract

import pytest

from ember.module import Context as ModuleContext
from ember.module import Module


@dataclass(frozen=True)
class Increment(Module[int, int]):
    """A stateful extension implements only its typed forward behavior."""

    amount: int

    def forward(self, inputs: int) -> int:
        return inputs + self.amount


class Compose(Module[int, int]):
    """Compose existing modules through their public dispatch method."""

    def __init__(self, *, first: Module[int, int], second: Module[int, int]) -> None:
        """Retain typed component state independently of each call result."""
        self.first: Module[int, int] = first
        self.second: Module[int, int] = second

    def forward(self, inputs: int) -> int:
        return self.second.run(self.first(inputs))


def test_module_dispatch_and_composition_preserve_component_state() -> None:
    first: Increment = Increment(amount=2)
    second: Increment = Increment(amount=3)
    module: Compose = Compose(first=first, second=second)
    assert isabstract(Module)
    assert repr(module) == "Compose()"
    assert first(4) == first.run(4) == first.forward(4) == 6
    assert module(4) == module.run(4) == module.forward(4) == 9
    assert first.amount == 2 and second.amount == 3


def test_context_nested_scopes_restore_after_exceptions() -> None:
    context: ModuleContext[int] = ModuleContext(value=1)
    value: int
    assert context.value == 1 and repr(context) == "Context(value=1)"
    with context.override(2) as value:
        assert value == context.value == 2
        with pytest.raises(ValueError, match="scope failed"):
            with context.override(3):
                assert context.value == 3
                raise ValueError("scope failed")
        assert context.value == 2
    assert context.value == 1
    with context.override(4):
        assert context.value == 4
    assert context.value == 1


def test_context_instances_and_copied_bindings_are_independent() -> None:
    first: ModuleContext[int] = ModuleContext(value=1)
    second: ModuleContext[int] = ModuleContext(value=2)
    copied: Context = copy_context()
    with first.override(3):
        assert first.value == 3 and second.value == 2
        assert copied.run(lambda: first.value) == 1
        with second.override(4):
            assert first.value == 3 and second.value == 4
        assert second.value == 2
    assert first.value == 1


def test_context_binding_is_not_inherited_by_new_threads() -> None:
    context: ModuleContext[int] = ModuleContext(value=1)
    executor: ThreadPoolExecutor
    with context.override(2), ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(lambda: context.value).result() == 1
        assert context.value == 2
    assert context.value == 1


async def task_value(context: ModuleContext[int]) -> int:
    """Let the caller change its own binding before reading the task's binding."""
    await asyncio.sleep(0)
    return context.value


async def isolated_tasks(context: ModuleContext[int]) -> tuple[int, int]:
    """Child tasks inherit their creation scope without sharing later bindings."""
    first: asyncio.Task[int]
    second: asyncio.Task[int]
    with context.override(2):
        first = asyncio.create_task(task_value(context))
    with context.override(3):
        second = asyncio.create_task(task_value(context))
    return await first, await second


def test_context_tasks_keep_their_creation_binding() -> None:
    context: ModuleContext[int] = ModuleContext(value=1)
    assert asyncio.run(isolated_tasks(context)) == (2, 3)
    assert context.value == 1
