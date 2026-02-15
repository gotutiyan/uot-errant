from base import ExpBase
from cleme import CLEME
from gec_metrics import get_metric
from gec_metrics.meta_eval import MetaEvalSEEDA
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import asdict
from seeda import ExpSEEDA
from dataclasses import dataclass

class ExpScatter(ExpBase):
    @dataclass
    class Config(ExpBase.Config):
        ref: str = 'ne-fluency'

    def __init__(self, config=None):
        super().__init__(config)
        self.meta = MetaEvalSEEDA(MetaEvalSEEDA.Config('+fluency'))
        self.ref_loader = ExpSEEDA()
        self.meta.system_data['references'] = self.ref_loader.load_refs(self.config.ref)
        plt.rcParams["font.size"] = 15
    
    def run(self, metric):
        if self.exists(f"{metric.__class__.__name__}.json"):
            res = self.load_json(f"{metric.__class__.__name__}.json")
        else:
            res = self.meta.corr_system(metric, aggregation='trueskill').ts_edit
            res = {
                'human_scores': res.human_scores,
                'metric_scores': res.metric_scores,
                'models': self.meta.system_data['models']
            }
        return res

    def plot(self, res, is_text=False, **args):
        human_scores = res['human_scores']
        metric_scores = res['metric_scores']
        plt.scatter(
            human_scores, metric_scores,
            s=50,
            **args,
        )
        if is_text:
            models = self.meta.system_data['models']
            for i in range(len(human_scores)):
                if models[i] not in ['REF-F', 'GPT-3.5', 'REF-M']:
                    continue
                plt.text(
                    human_scores[i], metric_scores[i], models[i]
                )
        plt.xlabel('Human score')
        plt.ylabel('Metric score')
        plt.grid(alpha=0.5, axis='y')
        return res
        
import argparse

def main(args):
    exp = ExpScatter()
    metrics = [
        get_metric('errant')(),
        get_metric('pterrant')(),
        CLEME(),
        exp.uot_errant
    ]
    for m in metrics:
        print(f"\nMetric{m=}")
        path = m.__class__.__name__ + '.json'
        if exp.exists(path):
            out = exp.load_json(path)
        else:
            out = exp.run(m)
            exp.save_json(out, path)
        label = m.__class__.__name__
        if label != 'ERRANT':
            label = label.replace('ERRANT', '-ERRANT')
        exp.plot(
            out,
            is_text=path == 'CLEME.json',
            label=label,
        )
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.tight_layout()
    plt.savefig(exp.base_path / 'scatter.png')
    plt.savefig(exp.base_path / 'scatter.pdf')
        

def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input', required=True)
    # parser.add_argument('--output', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)