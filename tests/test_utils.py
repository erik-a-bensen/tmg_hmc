import pytest 
import numpy as np
from hypothesis import given, strategies as st

from tmg_hmc.utils import compiled_library_available, arccos

def test_compiled_library_installed():
    assert compiled_library_available()

@given(st.floats(-1, 1))
def test_arccos_identity(x):
    result = arccos(x)
    expected = np.arccos(x)
    assert np.isclose(result, expected, atol=1e-7)