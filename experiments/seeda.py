from base import ExpBase
from dataclasses import dataclass, asdict
from gec_metrics.meta_eval import MetaEvalSEEDA
from gec_metrics.metrics import ERRANT, PTERRANT, GLEU, GREEN
from uot_errant import UOTERRANT
from gec_metrics import get_metric
from cleme import CLEME
from gecommon import Parallel
from pathlib import Path
from gec_datasets import GECDatasets
import subprocess
from tqdm import tqdm


class ExpSEEDA(ExpBase):
    def __init__(self, config=None):
        super().__init__(config)
        self.gec = GECDatasets('exp-datasets')
        self.data_path = self.gec.base_path / 'conll14'
        self.meta = MetaEvalSEEDA()
        self.meta_flu = MetaEvalSEEDA(MetaEvalSEEDA.Config('+fluency'))

        srcs_full = open("meta_eval_data/SEEDA/outputs/all/INPUT.txt").read().rstrip().split('\n')
        srcs_subset = open("meta_eval_data/SEEDA/outputs/subset/INPUT.txt").read().rstrip().split('\n')
        self.subset_indices = [i for i, s in enumerate(srcs_full) if s in srcs_subset]

        self.errant = ERRANT()
        self.pterrant = PTERRANT()
        # self.pterrant_rem = PTERRANTRemove()
        self.cleme = CLEME(CLEME.Config(mode='independent'))
        self.gleu = GLEU()
        self.green = GREEN()
        # self.uot_errant_st = UOTERRANTSentenceTransformer(self.uot_errant.config)
    
    def load_refs(self, data_id):
        if data_id == 'official':
            refs = self.gec.load('conll14').refs
            return [self.get_subset(r) for r in refs]
        if data_id == '10refs':
            path = self.data_path / '10refs'
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    f"wget -P {str(path)} https://aclanthology.org/attachments/P15-1068.Datasets.zip".split(' ')
                )
                subprocess.run(
                    f"unzip {str(path)}/P15-1068.Datasets.zip -d {str(path)}".split(' ')
                )
            refs = []
            for i in range(1, 11):
                sents = Parallel.from_m2(
                    str(self.data_path / '10refs' / f'10gec_annotations/A{i}.m2')
                ).trgs
                refs.append(self.get_subset(sents))
            return refs
        elif data_id == 'e-fluency':
            path = self.data_path / 'reassess-gec'
            if not path.exists():
                subprocess.run(f'git clone https://github.com/keisks/reassess-gec.git {path}'.split(' '))
            paths = [
                # Note: expert_fluencyA is used as a system in SEEDA
                path / 'annotations/expert_annotations/expert_fluency/expert_fluencyB'
            ]
        elif data_id == 'e-minimal':
            path = self.data_path / 'reassess-gec'
            if not path.exists():
                subprocess.run(f'git clone https://github.com/keisks/reassess-gec.git {path}'.split(' '))
            paths = [
                # Note: expert_minimalA is used as a system in SEEDA
                path / 'annotations/expert_annotations/expert_minimal/expert_minimalB'
            ]
        elif data_id == 'ne-fluency':
            path = self.data_path / 'reassess-gec'
            if not path.exists():
                subprocess.run(f'git clone https://github.com/keisks/reassess-gec.git {path}'.split(' '))
            paths = [
                path / 'annotations/turker_annotations/turkers_fluency/turker_fluencyA',
                path / 'annotations/turker_annotations/turkers_fluency/turker_fluencyB'
            ]
        elif data_id == 'ne-minimal':
            path = self.data_path / 'reassess-gec'
            if not path.exists():
                subprocess.run(f'git clone https://github.com/keisks/reassess-gec.git {path}'.split(' '))
            paths = [
                path / 'annotations/turker_annotations/turkers_minimal/turker_minimalA',
                path / 'annotations/turker_annotations/turkers_minimal/turker_minimalB'
            ]
        refs = [p.read_text().rstrip().split('\n') for p in paths]
        refs = [self.get_subset(r) for r in refs]
        return refs

    def get_subset(self, sentences):
        return [sentences[i] for i in self.subset_indices]

    def run(self):
        # metric_cls = get_metric('llmkobayashi24hfsent')
        # metric = [metric_cls(metric_cls.Config(
        #     model='google/gemma-3-12b-it',  # The model name or path for a language model.
        # ))]
        for metric in [self.gleu, self.green, self.errant, self.pterrant, self.pterrant_rem, self.cleme, self.uot_errant]:
        # for metric in [self.uot_errant]:
            path = f'{metric.__class__.__name__}.json'
            if isinstance(metric, (UOTERRANT)):
                path = f'{metric.__class__.__name__}-{metric.config.vectorize}-{metric.config.cost}-{metric.config.mass}-{metric.config.reg}-{metric.config.reg_m}.json'
            if not (self.base_path / path).exists():
                raw_results = dict()
                for ref_id in tqdm(['official', '10refs', 'e-minimal', 'ne-minimal']):
                    self.meta.system_data['references'] = self.load_refs(ref_id)
                    results = self.meta.corr_system(metric, aggregation='trueskill')
                    raw_results[ref_id] = asdict(results)
                for ref_id in tqdm(['e-fluency', 'ne-fluency']):
                    self.meta_flu.system_data['references'] = self.load_refs(ref_id)
                    results = self.meta_flu.corr_system(metric, aggregation='trueskill')
                    raw_results[ref_id] = asdict(results)
                self.save_json(raw_results, path)
            else:
                raw_results = self.load_json(path)
            self.latexfy(raw_results, title=metric.__class__.__name__)
        return
            
    def latexfy(self, raw_results, title='Metric'):
        aspect = 'ts_edit'
        ref_list = ['official', '10refs', 'e-minimal', 'ne-minimal', 'e-fluency', 'ne-fluency']
        results = sorted(raw_results.items(), key=lambda x: ref_list.index(x[0]))
        line = []
        for name, res in results:
            line.append(res[aspect]['pearson'])
            line.append(res[aspect]['spearman'])
            
        line = list(map(lambda x: f"{x:.3f}", line))
        latex_str = ' & '.join([title] + line) + '\\\\'
        latex_str = latex_str.replace('0.', '.')
        print(latex_str)
        return latex_str
        

if __name__ == '__main__':
    exp = ExpSEEDA()
    exp.run()
