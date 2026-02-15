from base import ExpBase
from gec_metrics.meta_eval import MetaEvalSEEDA
from gec_metrics import get_metric
from gec_metrics.metrics import inputs_handler
from gecommon import CachedERRANT
from tqdm import tqdm
from cleme import CLEME
import matplotlib.pyplot as plt

class ExpEfficiency(ExpBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.meta = MetaEvalSEEDA(MetaEvalSEEDA.Config('+fluency'))
        self.cleme = CLEME(CLEME.Config(mode='independent'))
        self.errant = CachedERRANT()
    
    def run(self, metric_id: str = 'errant'):
        metric = None
        if metric_id == 'uot-errant':
            metric = self.uot_errant
        elif metric_id == 'cleme':
            metric = self.cleme
        else:
            metric = get_metric(metric_id)()
        
        data = self.meta.load_system_data()
        hyps = data['hypotheses']
        names = data['models']
        srcs = data['sources']
        refs = data['references']
        for sys_id, hyp in tqdm(enumerate(hyps), total=len(hyps)):
            edits = [self.errant.extract_edits(s, h) for s, h in zip(srcs, hyp)]
            flatten_edits = [e for ee in edits for e in ee]
            num_edits = len(flatten_edits)
            if hasattr(metric, 'cache_parse'):
                metric.cache_parse = dict()
                metric.cache_annotate = dict()
            with self.timer(f'{names[sys_id]}'):
                score = metric.score_corpus(
                    **inputs_handler(metric, srcs, hyp, refs)
                )
        return {
            'timer': self.get_timer_results(),
            'num_edits': num_edits
        }

    def show(self):
        plt.rcParams["font.size"] = 15
        plt.figure(figsize=(10, 4))
        errant = CachedERRANT()
        metrics = [
            'errant',
            'gleu',
            'green',
            'impara',
            'some',
            'scribendi',
            'pterrant',
            'uot-errant',
            'cleme'
        ]
        data = self.meta.load_system_data()
        srcs = data['sources']
        refs = data['references']
        num_sents = len(srcs)
        
        models = data['models']
        num_edits = []
        for hyps in data['hypotheses']:
            num = 0
            for s, h in zip(srcs, hyps):
                num += len(errant.extract_edits(s, h))
                # num += len(h.split(' '))
            num_edits.append(num)
        models = [f"{models[i]} ({num_edits[i] / num_sents:.2f})" for i in range(len(models))]
        sorted_models = sorted(models, key=lambda x:num_edits[models.index(x)])
        sorted_num_edits = sorted(num_edits)
        num_edits_ratio = [e / sorted_num_edits[-1] for e in sorted_num_edits]
        print(sorted_models)
        print(num_edits_ratio)
        metric_names = {
            'errant': 'ERRANT',
            'pterrant': 'PT-ERRANT',
            'gleu': 'GLEU',
            'green': 'GREEN',
            'some': 'SOME',
            'impara': 'IMPARA',
            'scribendi': 'Scribendi',
            'cleme': 'CLEME',
            'uot-errant': 'UOT-ERRANT'
        }
        for metric_i, metric in enumerate(metrics):
            results = self.load_json(f"{metric}.json")
            times = [
                results['timer'][m.split(' ')[0]]
                for m in sorted_models
            ]
            times = [t / num_sents for t in times]
            plt.plot(
                num_edits_ratio,
                times,
                label=metric_names[metric],
                marker=[".", "o", "v", "x", "1", "2", "8", "s", "+", "D"][metric_i]
            )
        plt.legend()
        plt.tight_layout()
        plt.xticks(num_edits_ratio, sorted_models, rotation=-90)
        plt.ylabel('Computation time (seconds)')
        plt.xlabel('Models (Average number of edits per sentence)')
        plt.grid(alpha=0.5, axis='y')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5)
        plt.savefig(f'{self.base_path}/efficiency.png', bbox_inches='tight')
        plt.savefig(f'{self.base_path}/efficiency.pdf', bbox_inches='tight')

                

import argparse

def main(args):
    exp = ExpEfficiency()
    if args.show:
        exp.show()
        return
    # for metric_id in [
    #     'errant',
    #     'gleu',
    #     'green',
    #     'impara',
    #     'some',
    #     'scribendi',
    #     'pterrant',
    #     'uot-errant',
    #     'cleme'
    # ]:
        # results = exp.run(metric_id)
        # exp.save_json(results, metric_id + '.json')
    exp.show()
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    # parser.add_argument('--output', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)