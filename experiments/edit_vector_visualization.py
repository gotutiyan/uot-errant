from base import ExpBase
from gec_metrics.meta_eval import MetaEvalSEEDA
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import asdict
from gec_datasets import GECDatasets
from statistics import stdev, median, mean
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from collections import Counter

class ExpEditVectorVisualization(ExpBase):
    @dataclass
    class Config(ExpBase.Config):
        aspect: str = 'etype'
        reduction: str = 'tsne'
        cat: str = '2'

    def __init__(self, config=None):
        super().__init__(config)
        plt.rcParams["font.size"] = 15
    
    def run(self):
        gec = GECDatasets('exp-datasets')
        srcs = gec.load('conll14').srcs
        hyps = Path("meta_eval_data/SEEDA/outputs/all/GPT-3.5.txt").read_text().rstrip().split('\n')
        edits = [self.uot_errant.edit_extraction(s, h) for s, h in zip(srcs, hyps)]
        edit_vectors, mass = self.uot_errant.edit_vector(srcs, edits)
        
        flatten_edits = [ee for e in edits for ee in e]
        print(f'{len(flatten_edits)=}')
        if self.config.cat == '1':
            labels = [e.type[0] for e in flatten_edits]
        else:
            labels = [e.type[2:] for e in flatten_edits]
        flatten_vectors = np.array([ee for e in edit_vectors for ee in e])
        self.plot_embeddings_pca_2d(
            flatten_vectors, labels
        )
        return

    def plot_embeddings_pca_2d(self, embeddings: np.ndarray,
                            labels: np.ndarray = None,
                            title: str = "2D PCA of Embeddings",
                            figsize: tuple = (10, 9)):
        assert embeddings.ndim == 2
        assert embeddings.shape[0] > 0
        if self.config.reduction == 'pca':
            pca = PCA(n_components=2, random_state=42)
        elif self.config.reduction == 'tsne':
            pca = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=300, init='pca', learning_rate='auto')
        embeddings_2d = pca.fit_transform(embeddings)

        plt.figure(figsize=figsize)

        if labels is None:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, label="Data points")
            plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=1)
        else:
            assert embeddings.shape[0] == len(labels)

            label_freq = Counter(labels)
            unique_labels = sorted(list(set(labels)))
            n_unique_labels = len(unique_labels)
            n_unique_labels = len([k for k, v in label_freq.items() if v > 50])

            label_to_int = {label: i for i, label in enumerate(unique_labels)}
            int_to_label = {v: k for k, v in label_to_int.items()}
            numeric_labels_for_colors = np.array([label_to_int[l] for l in labels])
            numeric_unique_labels = np.array([label_to_int[l] for l in unique_labels])

            if n_unique_labels <= 10:
                colors = plt.get_cmap('tab10', n_unique_labels)
            elif n_unique_labels <= 20:
                colors = plt.get_cmap('tab20', n_unique_labels)
            else:
                # when so many labels exist
                colors = plt.get_cmap('viridis', n_unique_labels)

            color_id = 0
            for i, label_val in enumerate(numeric_unique_labels):
                if label_freq[int_to_label[label_val]] <= 50:
                    continue
                idx = np.where(numeric_labels_for_colors == label_val)
                plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                            color=colors(color_id),
                            label=unique_labels[label_val],
                            alpha=0.7)
                color_id += 1

            num_legend_cols = min(5, n_unique_labels)
            plt.legend(
                bbox_to_anchor=(0.5, -0.07),
                loc='upper center',
                ncol=num_legend_cols,
            )
        plt.axhline(0, color='grey', lw=0.5, linestyle='--')
        plt.axvline(0, color='grey', lw=0.5, linestyle='--')
        plt.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()
            
import argparse

def main(args):
    config = ExpEditVectorVisualization.Config(
        aspect='etype',
        reduction=args.reduction,
        cat=args.cat,
    )
    exp = ExpEditVectorVisualization(config)
    out = exp.run()
    exp.savefig(f'{config.model}-cat{args.cat}-{args.reduction}.png')
    exp.savefig(f'{config.model}-cat{args.cat}-{args.reduction}.pdf')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reduction', default='tsne')
    parser.add_argument('--cat', default=2)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)