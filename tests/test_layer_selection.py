"""Tests for layer-selection resolution."""

from __future__ import annotations

import pytest

from ablms.exceptions import UnsupportedOperationError
from ablms.utils.layers import resolve_layer_selection

# A 12-block model: 13 selectable indices, 0..12.
NUM_LAYERS = 12


def resolve(layer, **kwargs):
    return resolve_layer_selection(layer, NUM_LAYERS, model_name="testmodel", **kwargs)


class TestSingleLayer:
    """A single int passes through unchanged, preserving today's behaviour."""

    @pytest.mark.parametrize("layer", [-1, 0, 6, 12, -13])
    def test_valid_index_is_returned_unchanged(self, layer):
        assert resolve(layer) == layer

    def test_negative_index_is_not_normalized(self):
        """layer=-1 must stay -1 so EmbeddingOutput.layer reads as it does today."""
        assert resolve(-1) == -1

    @pytest.mark.parametrize("layer", [13, 99, -14])
    def test_out_of_range_raises(self, layer):
        with pytest.raises(ValueError, match="out of range"):
            resolve(layer)

    def test_error_names_the_model_and_the_valid_range(self):
        with pytest.raises(ValueError, match="testmodel"):
            resolve(13)
        with pytest.raises(ValueError, match=r"0\.\.12"):
            resolve(13)


class TestAllLayers:
    def test_all_returns_every_index_ascending(self):
        assert resolve("all") == list(range(13))

    def test_all_is_case_insensitive(self):
        assert resolve("ALL") == list(range(13))

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Invalid layer selection"):
            resolve("last")


class TestLayerList:
    def test_order_is_preserved(self):
        assert resolve([12, 0, 6]) == [12, 0, 6]

    def test_negatives_are_resolved(self):
        assert resolve([0, -1]) == [0, 12]

    def test_single_element_list_keeps_list_form(self):
        """The argument's type decides the output shape, not its length."""
        assert resolve([-1]) == [12]

    def test_tuple_and_range_are_accepted(self):
        assert resolve((0, 6)) == [0, 6]
        assert resolve(range(0, 13, 6)) == [0, 6, 12]

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            resolve([])

    def test_duplicates_after_resolution_raise(self):
        """[12, -1] is the form a caller writes by accident."""
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            resolve([12, -1])

    def test_literal_duplicates_raise(self):
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            resolve([6, 6])

    def test_out_of_range_member_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            resolve([0, 13])


class TestRejectedTypes:
    def test_bool_is_rejected(self):
        """bool subclasses int; True must not silently mean layer 1."""
        with pytest.raises(ValueError):
            resolve(True)

    def test_float_is_rejected(self):
        with pytest.raises(ValueError):
            resolve(1.0)

    def test_none_is_rejected(self):
        with pytest.raises(ValueError):
            resolve(None)


class TestFinalLayerOnlyModels:
    """Models like AbLang expose nothing but their final layer."""

    def restricted(self, layer):
        return resolve(layer, supports_intermediate_layers=False)

    @pytest.mark.parametrize("layer", [-1, 12])
    def test_final_layer_is_allowed(self, layer):
        assert self.restricted(layer) == layer

    def test_default_is_allowed(self):
        assert self.restricted(-1) == -1

    @pytest.mark.parametrize("layer", [0, 3, -2])
    def test_other_single_layers_raise(self, layer):
        with pytest.raises(UnsupportedOperationError, match="final layer"):
            self.restricted(layer)

    def test_all_raises(self):
        with pytest.raises(UnsupportedOperationError):
            self.restricted("all")

    def test_multi_element_list_raises(self):
        with pytest.raises(UnsupportedOperationError):
            self.restricted([0, 12])

    @pytest.mark.parametrize("layer", [[-1], [12]])
    def test_list_form_is_rejected_even_for_the_final_layer(self, layer):
        """A list requests a layer axis, even a single-element one naming the
        final layer. A model restricted to its final layer cannot produce a
        layer axis, so this must raise regardless of which index the list
        names.
        """
        with pytest.raises(UnsupportedOperationError, match="layer axis"):
            self.restricted(layer)
