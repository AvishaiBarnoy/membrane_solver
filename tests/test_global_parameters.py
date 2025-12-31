import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver


class DummyObj:
    def __init__(self, **options):
        self.options = dict(options)


def test_global_parameters_attribute_and_dict_access_are_consistent():
    params = GlobalParameters()

    params.set("volume_stiffness", 123.0)
    assert params.get("volume_stiffness") == 123.0
    assert params.volume_stiffness == 123.0

    params.volume_stiffness = 456.0
    assert params.volume_stiffness == 456.0
    assert params.get("volume_stiffness") == 456.0


def test_parameter_resolver_prefers_object_over_global():
    global_params = GlobalParameters()
    global_params.set("volume_stiffness", 10.0)
    resolver = ParameterResolver(global_params)

    body0 = DummyObj(volume_stiffness=2.0)
    body1 = DummyObj()

    assert resolver.get(body0, "volume_stiffness") == 2.0
    assert resolver.get(body1, "volume_stiffness") == 10.0
    assert resolver.get(None, "volume_stiffness") == 10.0


def test_parameter_resolver_unknown_key_returns_none_like_global_get():
    global_params = GlobalParameters()
    resolver = ParameterResolver(global_params)
    body = DummyObj()

    assert resolver.get(body, "does_not_exist") is None
