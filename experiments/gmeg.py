from gec_metrics.meta_eval import MetaEvalBase
from gec_metrics.meta_eval import MetaEvalSEEDA
from gec_metrics.metrics import MetricBase
from gec_metrics import get_metric
import subprocess
from dataclasses import dataclass, asdict
from gec_metrics.meta_eval.utils import read_lines
import glob
import os
from pathlib import Path
from base import ExpBase
from tqdm import tqdm
from cleme import CLEME
from gecommon import CachedERRANT
import matplotlib.pyplot as plt
from tabulate import tabulate

class MetaEvalGMEG(MetaEvalBase):
    MODELS = ['amu', 'lstm', 'lstm-r', 'marian', 'nus', 'transformer', 'ref0', 'source']
    SCORE_ID = ['score']
    @dataclass
    class Config(MetaEvalBase.Config):
        data: str = 'wiki'

    @dataclass
    class  GMEGSystemCorrOutput(MetaEvalBase.Output):
        '''The dataclass to store the meta-evaluation results.
        
        Args:
            ts (MetaEvalBase.Corr):
                The correlation using TrueSkill-based human evaluation.
            ts (MetaEvalBase.Corr):
                The correlation using Expected Wins-based human evaluation.
        '''
        corr: MetaEvalBase.Corr = None
        names: list[str] = None

    def __init__(self, config: MetaEvalBase.Config = None):
        super().__init__(config)
        self.system_data = self.load_system_data()
        self.sentence_data = self.load_sentence_data()
        self.cleme = CLEME(CLEME.Config(mode='independent'))
        self.errant = CachedERRANT()

    def download(self):
        subprocess.run(
            f'git clone https://github.com/grammarly/GMEG.git meta_eval_data/GMEG'.split(' ')
        )

    def load_system_data(self, split='test', data: str = 'wiki'):
        data_dir = glob.glob('**/meta_eval_data/GMEG/', recursive=True)
        if len(data_dir) == 0:
            self.download()
            data_dir = glob.glob('**/meta_eval_data/GMEG/', recursive=True)
            assert len(data_dir) > 0
        data_dir = data_dir[0]
        data = {
            'hypotheses': [],
            'references': [],
            'human_score': dict(),
            'models': self.MODELS,
            'sources': []
        }
        sentences = []
        for model in self.MODELS:
            sents = read_lines(os.path.join(data_dir, f'data/{split}/{self.config.data}/{model}'))
            sentences.append(sents)
        data['hypotheses'] = sentences
        data['sources'] = read_lines(os.path.join(data_dir, f'data/{split}/{self.config.data}/source'))
        # Load ref0 to ref3
        data['references'] = [
            read_lines(os.path.join(data_dir, f'data/{split}/{self.config.data}/ref{ref_id}'))
            for ref_id in range(1, 4)
        ]
        assert len(data['references']) == 3

        human_score_csv = Path(data_dir) / f'data/{split}/{self.config.data}-corpus-scores.csv'
        data['human_score']['score'] = self.load_human_score(
            human_score_csv, data['models'][:]
        )
        return data

    def load_human_score(self, csv: str, sys_names: list[str]):
        lines = Path(csv).read_text().strip().split('\n')
        lines = lines[1:]  # remove the header
        names = []
        scores = []
        sys_names[sys_names.index('ref0')] = 'ref'
        for line in lines:
            name, score = line.split(',')
            names.append(name)
            scores.append(float(score))
        reordered_scores = [scores[names.index(n)] for n in sys_names]
        return reordered_scores

    def load_sentence_data(self):
        return None

    def corr_system(
        self,
        metric: MetricBase,
        aggregation='default'
    ) -> "GJGSystemCorrOutput":
        '''Compute system-level correlations.

        Args:
            metric (MetricBase): The metric to be evaluated.

        Returns:
            GJGSystemCorrOutput: The correlations.
        '''
        corrs = super().corr_system(metric, aggregation=aggregation)
        return self.GMEGSystemCorrOutput(
            corr = corrs[0]
        )
    
    def corr_sentence(self, metric: MetricBase) -> "GJGSentenceCorrOutput":
        '''Compute sentence-level correlations.

        Args:
            metric (MetricBase): The metric to be evaluated.

        Returns:
            GJGSentenceCorrOutput: The correlations.
        '''
        corrs = super().corr_sentence(metric)
        return self.GMEGSentenceCorrOutput(
            corr=corrs[0]
        )

class ExpGMEG(ExpBase):
    @dataclass
    class Config(ExpBase.Config):
        data: str = 'wiki'

    def __init__(self, config=None):
        super().__init__(config)
        self.meta = MetaEvalGMEG(MetaEvalGMEG.Config(
            data=self.config.data
        ))
    
    def run(self, metric):
        corr = self.meta.corr_system(
            metric,
            aggregation='trueskill'
        )
        return {
            'names': self.meta.load_system_data()['models'],
            'corr': asdict(corr)
        }

    def show(self, results, label='aa'):
        corr = results['corr']['corr']
        pea = corr['pearson']
        spe = corr['spearman']
        plt.scatter(corr['human_scores'], corr['metric_scores'], label=label)
        print(f' & {pea:.3f} & {spe:.3f}'.replace('0.', '.'))
        return 

    def show_density(self):
        '''Check error density in both SEEDA and GMEG-Data, corresponding to Figure 2 in the paper.
        '''
        plt.rcParams["font.size"] = 15
        plt.figure(figsize=(10, 4))
        errant = CachedERRANT()
        def get_density(meta):
            data = meta.load_system_data()
            y = []
            labels = []
            for i, hyps in enumerate(data['hypotheses']):
                num = 0
                for s, h in zip(data['sources'], hyps):
                    num += len(errant.extract_edits(s, h))
                print(data['models'][i], num / len(data['sources']))
                y.append(num / len(data['sources']))
                pref = 'GMEG' if isinstance(meta, MetaEvalGMEG) else 'SEEDA'
                labels.append(f'{pref}: ' + data['models'][i])
            temp = sorted(list(zip(y, labels)), key=lambda x:x[0])
            sorted_y = [t[0] for t in temp]
            sorted_labels = [t[1] for t in temp]
            return sorted_y, sorted_labels
            
        print('=== GMEG ===')
        gmeg_y, gmeg_labels = get_density(self.meta)
        seeda = MetaEvalSEEDA(MetaEvalSEEDA.Config('+fluency'))
        seeda_y, seeda_labels = get_density(seeda)
        
        plt.grid(alpha=0.5, axis='y')
        plt.bar(
            list(range(len(gmeg_y + seeda_y))),
            gmeg_y + seeda_y,
            tick_label=gmeg_labels + seeda_labels,
            # color=['orange'] * len(gmeg_labels) + ['steelblue'] * len(seeda_labels),
            color=['orange'] * len(gmeg_labels) + ['steelblue'] * len(seeda_labels),
            hatch=['///'] * len(gmeg_labels) + ['...'] * len(seeda_labels)
        )
        plt.xticks(rotation=-80)
        plt.xlabel('Models')
        plt.ylabel('Number of edits per sentence')
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=2)
        plt.savefig(self.base_path / 'density.png', bbox_inches='tight')
        plt.savefig(self.base_path / 'density.pdf', bbox_inches='tight')

    def show_etype_dist(self):
        '''Check error type wise distribution in both SEEDA and GMEG-Data.
        This is for the analysis described in the paragraph starting with "In GMEG-Data ..." in Section 4.3.
        '''
        plt.rcParams["font.size"] = 15
        plt.figure(figsize=(10, 4))
        errant = CachedERRANT()
        def get_etype_stats(meta):
            count = dict()
            data = meta.load_system_data()
            srcs = data['sources']
            hyps_list = data['hypotheses']
            for hyps in hyps_list:
                for s, h in zip(srcs, hyps):
                    edits = errant.extract_edits(s, h)
                    for e in edits:
                        t = e.type[2:]
                        if t not in count:
                            count[t] = 1
                        else:
                            count[t] += 1
            return count
                
        print('=== GMEG ===')
        gmeg_count = get_etype_stats(self.meta)
        seeda = MetaEvalSEEDA(MetaEvalSEEDA.Config('+fluency'))
        seeda_count = get_etype_stats(seeda)
        table = []
        gmeg_total = sum(gmeg_count.values())
        seeda_total = sum(seeda_count.values())
        print(f"{gmeg_total=}")
        print(f"{seeda_total=}")
        for e in gmeg_count:
            table.append([
                e, 
                f"{gmeg_count[e]} ({gmeg_count[e] / gmeg_total})",
                f"{seeda_count[e]} ({seeda_count[e] / seeda_total})",
            ])
        print(tabulate(table, headers=['Etype', 'GMEG-count', 'SEEDA-count']))
        

import argparse

def main(args):
    exp = ExpGMEG(ExpGMEG.Config(
        data='wiki'
    ))
    if args.density:
        exp.show_density()
        return
    if args.distribution:
        exp.show_etype_dist()
        return
    for metric_id in tqdm([
        'errant',
        'gleu',
        'green',
        'impara',
        'some',
        'scribendi',
        'pterrant',
        'uot-errant',
        'cleme',
        'llm'
    ][:]):
        if metric_id == 'uot-errant':
            metric = exp.uot_errant
        elif metric_id == 'cleme':
            metric = CLEME(CLEME.Config(mode='independent'))
        elif metric_id == 'llm':
            metric_cls = get_metric('llmkobayashi24hfsent')
            metric = metric_cls(metric_cls.Config(
                model='Qwen/Qwen3-8B',  # The model name or path for a language model.
                # quantization='4bit'
            ))
        else:
            metric = get_metric(metric_id)()
        if args.show:
            results = exp.load_json(f'{exp.config.data}/{metric_id}.json')
            print(metric_id, end='')
            exp.show(results, label=metric_id)
            plt.legend()
            plt.xlabel('Human scores')
            plt.ylabel('Metric scores')
            plt.savefig(f'{exp.base_path}/scatter.png')
            plt.savefig(f'{exp.base_path}/scatter.pdf')
        else:
            results = exp.run(metric)
            exp.save_json(results, f'{metric.__class__.__name__}/{exp.config.data}/{metric_id}.json')
            exp.show(results, label=metric_id)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--density', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--distribution', action='store_true')
    # parser.add_argument('--output', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)