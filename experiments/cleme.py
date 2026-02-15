from gec_metrics.metrics import MetricBaseForReferenceBased
from CLEME.cleme.cleme import DependentChunkMetric, IndependentChunkMetric
from CLEME.cleme.data import M2DataReader
from CLEME.cleme.scorers.scorer_sentence import SentenceScorer
from CLEME.cleme.scorers.scorer_base import compute_f, compute_acc
from gecommon import CachedERRANT
from pathlib import Path
import subprocess
import pprint
from dataclasses import dataclass
import numpy as np

class SentenceWiseScorer(SentenceScorer):
    '''This scorer returns sentence-level scores.
    '''
    def __call__(self, scorer_inputs: list[list[dict[str, int]]]) -> dict:
        """ Calculate sentence-level Accuracy Score """
        total_f, total_p, total_r, total_acc = [], [], [], []
        for sample_result in scorer_inputs:
            best_f, best_p, best_r, best_acc = -1.0, -1.0, -1.0, -1.0
            for ref_result in sample_result:
                _tp, _fp, _fn, _tn = ref_result["tp"], ref_result["fp"], ref_result["fn"], ref_result["tn"]
                _p, _r, _f = compute_f(_tp, _fp, _fn)
                _acc = compute_acc(_tp, _fp, _fn, _tn)
                # print(_tp, _fp, _fn, _f)
                if (_f > best_f) or \
                        (_f == best_f and _p > best_p) or \
                        (_f == best_f and _p == best_p and _r > best_r) or \
                        (_f == best_f and _p == best_p and _r == best_r and _acc < best_acc):
                    best_f, best_p, best_r, best_acc = _f, _p, _r, _acc
            total_f.append(best_f)
            total_p.append(best_p)
            total_r.append(best_r)
            total_acc.append(best_acc)

        f, p, r, acc = total_f, total_p, total_r, total_acc
        return {
            "num_sample": len(scorer_inputs),
            'F': f,
            'P': p,
            'R': r,
            'Acc': acc,
        }


class CLEME(MetricBaseForReferenceBased):
    '''This is a wrapper to use the official CLEME based on the gec-metrics interface.
        gec-metrics: https://github.com/gotutiyan/gec-metrics
    '''
    @dataclass
    class Config(MetricBaseForReferenceBased.Config):
        mode: str = 'independent'

    def __init__(self, config=None):
        super().__init__(config)
        # Evaluate using CLEME_dependent
        if self.config.mode == 'dependent':
            config_dependent = {
                "tp": {"alpha": 10.0, "min_value": 1.0, "max_value": 10.0, "reverse": False},
                "fp": {"alpha": 10.0, "min_value": 0.25, "max_value": 10.0, "reverse": True},
                "fn": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": False},
            }
            self.cleme = DependentChunkMetric(scorer='sentence', weigher_config=config_dependent)
        elif self.config.mode == 'independent':
            config_dependent = {
                "tp": {"alpha": 10.0, "min_value": 2.50, "max_value": 10.0, "reverse": False},
                "fp": {"alpha": 10.0, "min_value": 0.25, "max_value": 1.0, "reverse": True},
                "fn": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": False},
            }
            self.cleme = IndependentChunkMetric(scorer='sentence', weigher_config=config_dependent)
        self.cleme.scorer = SentenceWiseScorer()  # Use the sentence-level scorer defined above.

    def score_sentence(self, sources, hypotheses, references):
        temp_dir = Path('temp-dir-CLEME')
        temp_dir.mkdir(exist_ok=True)
        src_path = temp_dir / 'src'
        hyp_path = temp_dir / 'hyp'
        src_path.write_text('\n'.join(sources))
        hyp_path.write_text('\n'.join(hypotheses))
        ref_all_paths = []
        for ref_id in range(len(references)):
            ref_path = temp_dir / f'ref{ref_id}'
            ref_path.write_text('\n'.join(references[ref_id]))
            ref_all_paths.append(str(ref_path))
        hyp_m2_path = temp_dir / 'hyp.m2'
        ref_m2_path = temp_dir / 'ref.m2'
        subprocess.run(f'uv run errant_parallel -orig {src_path} -cor {hyp_path} -out {hyp_m2_path}'.split())
        subprocess.run(f'uv run errant_parallel -orig {src_path} -cor {" ".join(ref_all_paths)} -out {ref_m2_path}'.split())
        data = M2DataReader()
        hyp_dataset = data.read(hyp_m2_path)
        ref_dataset = data.read(ref_m2_path)
        scores, results = self.cleme.evaluate(hyp_dataset, ref_dataset)
        return scores['F']