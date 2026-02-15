import pytest
from gecommon import CachedERRANT
from .uot_errant import UOTERRANT
import ot
import numpy as np

SRCS = [
    'This sentences contain gramamtical error .',
    'This is a sentence .',
    'This is no change .',
]
HYPS = [
    'This sentence contains a grammatical error .',
    'the sentence was corrected into completely different one .',
    'This is no change .',
]
REFS = [
    'This sentence contains a gramamtical error .',
    'dummy sentence .',
    'This is no change .',
]

cases = [
    ([4, 5, 0], [])
]


class TestUOTERRANT:
    def test_shape(self):
        uot_errant = UOTERRANT()
        errant = CachedERRANT()
        edits_h = [errant.extract_edits(s, h) for s, h in zip(SRCS, HYPS)]
        edit_vectors_h, mass_h = uot_errant.edit_vector(SRCS, edits_h)
        assert len(edit_vectors_h) == 3
        assert [len(e) for e in edit_vectors_h] == [4, 5, 0]
        assert len(mass_h) == 3
        assert [len(e) for e in mass_h] == [4, 5, 0]

        edits_r = [errant.extract_edits(s, h) for s, h in zip(SRCS, REFS)]
        edit_vectors_r, mass_r = uot_errant.edit_vector(SRCS, edits_r)
        assert len(edit_vectors_r) == 3
        assert [len(e) for e in edit_vectors_r] == [3, 2, 0]
        assert len(mass_r) == 3
        assert [len(e) for e in mass_r] == [3, 2, 0]

        for sample_id in range(2):  # No edit exists in 3rd example, so we skip it
            print(np.array(edit_vectors_h[sample_id]).shape)
            print(np.array(edit_vectors_r[sample_id]).shape)
            cost_matrix = ot.dist(
                np.array(edit_vectors_h[sample_id]),
                np.array(edit_vectors_r[sample_id]),
                metric='euclidean'
            )
            assert cost_matrix.shape == ([4,5,0][sample_id], [3,2,0][sample_id])
            transport_plan = uot_errant.transport_plan(
                a=mass_h[sample_id],
                b=mass_r[sample_id],
                cost_matrix=cost_matrix
            )
            assert transport_plan.shape == ([4,5,0][sample_id], [3,2,0][sample_id])