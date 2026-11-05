"""Keep the parser tensor storage consistent with the task's parameter count."""

from ember.playground.breakout.tensors import Tensors


def test_the_network_has_the_task_parameter_count() -> None:
    assert sum(tensor.size for tensor in Tensors.zeros().tensors().values()) == 20_486
