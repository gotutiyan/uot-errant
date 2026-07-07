from dataclasses import dataclass, asdict
import abc
import json
from pathlib import Path
from uot_errant import UOTERRANT
import itertools
import matplotlib.pyplot as plt
from typing import Any
from dataclasses import is_dataclass, asdict
from gec_datasets import GECDatasets
import subprocess
from gecommon import Parallel
from timer import StopwatchDict

class ExpBase(abc.ABC):
    @dataclass
    class Config:
        model: str = 'google/electra-base-discriminator'
        base_path: str = 'exp-outputs'
        cost: str = 'euclidean'
        mass: str = 'norm'
        vectorize: str = 'ref-remove'
        reg: float = 0.1
        reg_m: float = 0.1

    def __init__(self, config=None):
        self.config = config if config is not None else self.Config()
        self.current_time = 0
        self.base_path = Path(self.config.base_path) / self.__class__.__name__
        self.save_json(self.config.__dict__, 'config.json')
        metric_cls = UOTERRANT
        self.uot_errant = metric_cls(metric_cls.Config(
            model=self.config.model,
            cost=self.config.cost,
            mass=self.config.mass,
            vectorize=self.config.vectorize,
            reg=self.config.reg,
            reg_m=self.config.reg_m
        ))
        self.timer = StopwatchDict()

    def get_timer_results(self):
        return self.timer.elapsed_time

    def get_config_str(self):
        strs = [f"{k}-{v}" for k, v in asdict(self.config).items() if isinstance(v, [str, int, float])]
        return '@@'.join(strs)

    def convert_dataclass(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return self.convert_dataclass(asdict(obj))
        elif isinstance(obj, dict):
            return {key: self.convert_dataclass(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_dataclass(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_dataclass(item) for item in obj)
        else:
            return obj

    def save_json(self, obj, name=None):
        if not name.endswith('.json'):
            name += '.json'
        path = self.base_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.convert_dataclass(obj), path.open('w'), indent=2)

    def load_json(self, name):
        path = self.base_path / name
        if not self.exists(name):
            return None
        data = json.load(path.open())
        return data
    
    def exists(self, name):
        path = self.base_path / name
        return path.exists()

    def savefig(self, name):
        path = self.base_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)

class ReferenceLoader():
    def __init__(self, data_path='exp-datasets'):
        self.data_path = Path(data_path)
        self.gec = GECDatasets(data_path)
        srcs_full = open("meta_eval_data/SEEDA/outputs/all/INPUT.txt").read().rstrip().split('\n')
        srcs_subset = open("meta_eval_data/SEEDA/outputs/subset/INPUT.txt").read().rstrip().split('\n')
        self.subset_indices = [i for i, s in enumerate(srcs_full) if s in srcs_subset]
        
    def load(self, data_id):
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
                path / 'annotations/expert_annotations/expert_fluency/expert_fluencyB'
            ]
        elif data_id == 'e-minimal':
            path = self.data_path / 'reassess-gec'
            if not path.exists():
                subprocess.run(f'git clone https://github.com/keisks/reassess-gec.git {path}'.split(' '))
            paths = [
                # path / 'annotations/expert_annotations/expert_minimal/expert_minimalA',
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