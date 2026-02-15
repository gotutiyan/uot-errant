from base import ExpBase
from gec_metrics.metrics import inputs_handler
from gec_metrics import get_metric
from uot_errant import UOTERRANT
from gec_metrics.meta_eval import MetaEvalSEEDA
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import asdict
import itertools
import pandas as pd
import seaborn as sns
from seeda import ExpSEEDA

class MetaEvalSEEDANew(MetaEvalSEEDA):
    def pairwise_analysis(
        self,
        metric
    ):
        '''Compute sentence-level correlations.

        Args:
            metric (MetricBase): The metric to be evaluated.

        Returns:
            **SentenceCorrOutput: The sentence-level correlations output.
        '''
        data = self.sentence_data
        pairwise_score = metric.score_pairwise(
            **inputs_handler(
                metric, data['sources'], data['hypotheses'], data['references']
            ),
        )  # (num_sentence, num_systems, num_systems)
        num_sents = len(data['sources'])
        num_sys = len(data['models'])
        stats = dict()
        for name in sorted(list(data['human_score'].keys())):
            human_scores = data['human_score'][name]
            denominator = 0
            stats[name] = stats.get(name, dict())
            for src_id in range(num_sents):
                for annotate_id in range(len(human_scores[src_id])):
                    for sys1, sys2 in itertools.combinations(range(num_sys), 2):
                        # The human score is minus ranking value,
                        #   so higher values indicate higher quality.
                        h1 = human_scores[src_id][annotate_id][sys1]
                        h2 = human_scores[src_id][annotate_id][sys2]
                        if None in [h1, h2]:
                            continue
                        if h1 == h2:
                            continue
                        denominator += 1
                        human_judge = 1 if h1 > h2 else -1
                        key = tuple(sorted([sys1, sys2]))
                        stats[name][key] = stats[name].get(key, {'agree': 0, 'not-agree': 0})
                        # SEEDA considers metric's tie result a loss.
                        metric_judge = pairwise_score[src_id][sys1][sys2]
                        if metric_judge == 0:
                            metric_judge = -1
                        if metric_judge == human_judge:
                            stats[name][key]['agree'] += 1
                        else:
                            stats[name][key]['not-agree'] += 1
            
            stats[name] = {
                k: stats[name][k]['agree'] / (stats[name][k]['agree'] + stats[name][k]['not-agree'])
                for k in stats[name]
            }
            # sort by the span index
            stats[name] = sorted(stats[name].items(), key=lambda x: x[0])
        return stats
    
    def pairwise_analysis_plot(
        self,
        results: list[tuple, float]
    ):
        plt.figure(figsize=(20, 16))
        x_vals = [pair[0][0] for pair in results]  # rank A
        y_vals = [pair[0][1] for pair in results]  # rank B
        z_vals = [pair[1] for pair in results]  # accuracy
        df = pd.DataFrame({"x": x_vals, "y": y_vals, "z": z_vals})
        heatmap_data = df.pivot(index="y", columns="x", values="z")
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            cmap="coolwarm",
            center=0,
            cbar=True,
            fmt=".2f",
            annot_kws={"size": 20, "weight": "bold"}
        )
        models = self.sentence_data['models']
        ax.set_xticklabels(models[:-1], fontsize=35, fontweight="bold")
        ax.set_yticklabels(models[1:], fontsize=35, fontweight="bold")
        # ax.set_xlabel("", fontsize=35, fontweight='bold')
        # ax.set_ylabel("", fontsize=35, fontweight='bold')
        # ax.xaxis.set_label_position('top') 
        # ax.xaxis.tick_top()
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45, va='top')

        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(labelsize=35)
        for label in cbar.ax.get_yticklabels():
            label.set_fontsize(35)
            label.set_fontweight("bold")
        return ax.get_figure()
        

class ExpPairwiseAnalysis(ExpBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.ref_loader = ExpSEEDA()
        self.meta = MetaEvalSEEDANew(MetaEvalSEEDANew.Config(
            '+fluency'
        ))
        self.meta.system_data['references'] = self.ref_loader.load_refs('ne-fluency')
        self.meta.sentence_data['references'] = self.ref_loader.load_refs('ne-fluency')

    def run(self, metric):
        meta_out = self.meta.pairwise_analysis(metric)
        return meta_out

    def plot(self, out):
        fig = self.meta.pairwise_analysis_plot(out)
        return fig
        
import argparse

def main(args):
    exp = ExpPairwiseAnalysis()

    # Calculate system-level pairwise agreement rates
    for metric_id in ['errant', 'uot_errant', 'pterrant']:
    # for metric_id in ['pterrant']:
        metric = get_metric(metric_id)() if metric_id != 'uot_errant' else exp.uot_errant
        path = f"{metric.__class__.__name__}.json"
        exp = ExpPairwiseAnalysis()
        out = exp.run(metric) if not exp.exists(path) else exp.load_json(path)
        exp.plot(out['edit'])  # use agreements with edit-level human evaluation
        exp.save_json(out, path)
        plt.tight_layout()
        plt.savefig(exp.base_path / f'{metric_id}.png')
        plt.savefig(exp.base_path / f'{metric_id}.pdf')

    # Calculate and plot difference of agreement rate between metric 1 and metric 2
    for metric1, metric2 in [
        ['ERRANT', 'PTERRANT'],
        ['ERRANT', 'UOTERRANT']
    ]:
        # Load pre-calculated results
        out1 = exp.load_json(f"{metric1}.json")['edit']
        out2 = exp.load_json(f"{metric2}.json")['edit']
        diff = [] 
        assert len(out1) == len(out2)
        for elem2 in out2:
            for elem1 in out1:
                if elem1[0] == elem2[0]:
                    diff.append((elem2[0], elem2[1]- elem1[1]))
                    break
        exp.plot(diff)
        plt.tight_layout() 
        plt.savefig(exp.base_path / f'{metric2}-{metric1}.png')
        plt.savefig(exp.base_path / f'{metric2}-{metric1}.pdf')

def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input', required=True)
    # parser.add_argument('--output', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)
