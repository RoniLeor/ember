"""Typed callable modules and instance-owned execution contexts."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")
Value = TypeVar("Value")


class Module(ABC, Generic[Input, Output]):
    """Keep call dispatch separate from the component's forward behavior."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __call__(self, inputs: Input) -> Output:
        return self.run(inputs)

    def run(self, inputs: Input) -> Output:
        """Run one typed request and return its independent result."""
        return self.forward(inputs)

    @abstractmethod
    def forward(self, inputs: Input) -> Output:
        """Implement the component's behavior without replacing call dispatch."""
        ...


class Context(Generic[Value]):
    """Scope configuration bindings; use immutable values for isolation."""

    def __init__(self, *, value: Value) -> None:
        """Create a distinct context binding with the supplied default value."""
        self._value: ContextVar[Value] = ContextVar("ember.context", default=value)

    def __repr__(self) -> str:
        return f"Context(value={self.value!r})"

    @property
    def value(self) -> Value:
        """Read this execution context's current binding."""
        return self._value.get()

    @contextmanager
    def override(self, value: Value) -> Generator[Value, None, None]:
        """Restore the previous binding when the scope exits, including on errors."""
        token: Token[Value] = self._value.set(value)
        try:
            yield value
        finally:
            self._value.reset(token)
