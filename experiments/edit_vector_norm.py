from base import ExpBase
from gec_metrics.meta_eval import MetaEvalSEEDA
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import asdict
from gec_datasets import GECDatasets
from statistics import stdev, median, mean
from dataclasses import dataclass

class ExpNorm(ExpBase):
    @dataclass
    class Config(ExpBase.Config):
        aspect: str = 'etype'

    def __init__(self, config=None):
        super().__init__(config)
        plt.rcParams["font.size"] = 15
    
    def run(self):
        gec = GECDatasets('exp-datasets')
        srcs = gec.load('conll14').srcs
        hyps = Path("meta_eval_data/SEEDA/outputs/all/GPT-3.5.txt").read_text().rstrip().split('\n')
        edits = [self.uot_errant.edit_extraction(s, h) for s, h in zip(srcs, hyps)]
        edit_vectors, mass = self.uot_errant.edit_vector(srcs, edits)
        
        type2mass = dict()
        for sent_id in range(len(srcs)):
            for edit_id in range(len(edits[sent_id])):
                e = edits[sent_id][edit_id]
                if self.config.aspect == 'length':
                    t = e.o_end - e.o_start
                elif self.config.aspect == 'etype':
                    t = edits[sent_id][edit_id].type
                    t = t[2:]
                if t not in type2mass:
                    type2mass[t] = []
                type2mass[t].append(float(mass[sent_id][edit_id]))
        return type2mass

    def show(self, type2mass):
        for t, m in sorted(type2mass.items(), key=lambda x: sum(x[1])/len(x[1])):
            if len(m) >= 50:
                print(f"{t:10}: mean:{mean(m):.3f} +- {stdev(m):.3f} (Freq. {len(m)})")

    def latexify(self, type2mass):
        for t, m in sorted(type2mass.items(), key=lambda x: sum(x[1])/len(x[1])):
            if len(m) >= 50:
                print(f"{t:10} & {mean(m):.3f} $_\\text{{± {stdev(m):.3f}}}$ \\\\")
        
            
import argparse

def main(args):
    config = ExpNorm.Config(
        aspect='etype'
    )
    exp = ExpNorm(config)
    if exp.exists('data.json'):
        out = exp.load_json('data.json')
    else:
        out = exp.run()
        exp.save_json(out, 'data.json')
    # exp.show(out)
    exp.latexify(out)

def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input', required=True)
    # parser.add_argument('--output', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)