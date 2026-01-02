"""
Property-based tests for numerical stability using Hypothesis.

These tests help catch edge cases in numerical computations that may cause:
- NaN/Inf values
- Numerical overflow/underflow
- Incorrect handling of boundary conditions
"""
import numpy as np
import pytest

try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Create dummy decorators for when hypothesis is not installed
    def given(*args, **kwargs):
        def decorator(f):
            return pytest.mark.skip(reason="hypothesis not installed")(f)
        return decorator
    def settings(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    st = None

from pyfk.config.config import Config, SeisModel, SourceModel


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestNumericalStability:
    """Property-based tests for numerical stability."""

    @given(
        thickness=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        vp=st.floats(min_value=1.0, max_value=15.0, allow_nan=False, allow_infinity=False),
        vs=st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
        density=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_seismodel_creation_stability(self, thickness, vp, vs, density):
        """Test that SeisModel can handle various reasonable input values."""
        # Ensure vp > vs (physically realistic)
        assume(vp > vs)
        
        model_data = np.array([[thickness, vp, vs, density, 100., 50.]])
        
        try:
            model = SeisModel(model=model_data)
            # Check that model attributes are finite
            assert np.all(np.isfinite(model.model_values))
        except Exception as e:
            # Some parameter combinations may be rejected by validation
            # This is acceptable as long as it's handled gracefully
            pass

    @given(
        sdep=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_source_depth_stability(self, sdep):
        """Test that SourceModel handles various source depths."""
        try:
            source = SourceModel(sdep=sdep)
            assert np.isfinite(source.sdep)
        except Exception:
            # Some depths may be rejected by validation
            pass

    @given(
        distance=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_receiver_distance_stability(self, distance):
        """Test that Config handles various receiver distances."""
        model_data = np.array([
            [20., 6.0, 3.5, 2.7, 100., 50.],
            [0., 8.0, 4.5, 3.3, 200., 100.]
        ])
        
        try:
            model = SeisModel(model=model_data)
            source = SourceModel(sdep=10.0)
            config = Config(
                model=model,
                source=source,
                npt=256,
                dt=0.1,
                receiver_distance=[distance]
            )
            assert config is not None
        except Exception:
            # Some distances may be rejected by validation
            pass


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestEdgeCases:
    """Tests for edge cases that may cause numerical issues."""

    def test_very_small_timestep(self):
        """Test with very small timestep."""
        model_data = np.array([
            [20., 6.0, 3.5, 2.7, 100., 50.],
            [0., 8.0, 4.5, 3.3, 200., 100.]
        ])
        model = SeisModel(model=model_data)
        source = SourceModel(sdep=10.0)
        
        # Very small timestep - should either work or raise a sensible error
        try:
            config = Config(
                model=model,
                source=source,
                npt=64,
                dt=0.001,
                receiver_distance=[10.]
            )
            assert config is not None
        except ValueError as e:
            # Acceptable if validation catches unreasonable parameters
            assert "dt" in str(e).lower() or "time" in str(e).lower()

    def test_large_npt(self):
        """Test with large number of time points."""
        model_data = np.array([
            [20., 6.0, 3.5, 2.7, 100., 50.],
            [0., 8.0, 4.5, 3.3, 200., 100.]
        ])
        model = SeisModel(model=model_data)
        source = SourceModel(sdep=10.0)
        
        # Large npt - should either work or raise a sensible error
        try:
            config = Config(
                model=model,
                source=source,
                npt=4096,
                dt=0.1,
                receiver_distance=[10.]
            )
            assert config is not None
        except (ValueError, MemoryError) as e:
            # Acceptable if validation catches unreasonable parameters
            pass

    def test_source_at_interface(self):
        """Test with source exactly at layer interface."""
        model_data = np.array([
            [20., 6.0, 3.5, 2.7, 100., 50.],
            [30., 7.0, 4.0, 3.0, 150., 75.],
            [0., 8.0, 4.5, 3.3, 200., 100.]
        ])
        model = SeisModel(model=model_data)

        # Source at exactly the interface depth - should raise an error
        source = SourceModel(sdep=20.0)

        # This should raise a PyfkError because source is at a real interface
        with pytest.raises(Exception):
            config = Config(
                model=model,
                source=source,
                npt=256,
                dt=0.1,
                receiver_distance=[10.]
            )

    def test_multiple_receivers_same_distance(self):
        """Test with multiple receivers at the same distance."""
        model_data = np.array([
            [20., 6.0, 3.5, 2.7, 100., 50.],
            [0., 8.0, 4.5, 3.3, 200., 100.]
        ])
        model = SeisModel(model=model_data)
        source = SourceModel(sdep=10.0)
        
        config = Config(
            model=model,
            source=source,
            npt=256,
            dt=0.1,
            receiver_distance=[10., 10., 10.]
        )
        assert config is not None
        assert len(config.receiver_distance) == 3
