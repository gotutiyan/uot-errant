import streamlit as st
import argparse
from pathlib import Path
from gec_datasets import GECDatasets
from uot_errant import UOTERRANT
import matplotlib.pyplot as plt
import seaborn as sns
import ot
import numpy as np

def main(args):
    plt.rcParams["font.size"] = 20
    gec = GECDatasets('exp-datasets')
    hyps = Path(args.hyp).read_text().rstrip().split('\n')
    srcs = gec.load('conll14').srcs
    refs = open("/cl/nldata/GEC/conll14/reassess-gec/annotations/expert_annotations/expert_fluency/expert_fluencyB").read().rstrip().split('\n')
    
    uot_errant = UOTERRANT()
    srcs = srcs[:]
    hyps = hyps[:]
    refs = refs[:]
    num_sents = len(srcs)
    # num_sents = 100
    num_refs = len(refs)
    scores = []  # The shape will be: (num_sents, num_refs, )
    hyp_edits = [
        uot_errant.edit_extraction(
            srcs[sent_id],
            hyps[sent_id]
        ) for sent_id in range(num_sents)
    ]
    ref_edits = [
        uot_errant.edit_extraction(
            srcs[sent_id],
            refs[sent_id]
        ) for sent_id in range(num_sents)
    ]
    hyp_vectors, hyp_mass = uot_errant.edit_vector(srcs, hyp_edits)
    ref_vectors, ref_mass = uot_errant.edit_vector(srcs, ref_edits)
    # for sent_id in range(100):  # Show only first 10 samples
    for sent_id in range(num_sents):  # Show all samples, but takes a long time
        s = srcs[sent_id]
        h = hyps[sent_id]
        r = refs[sent_id]
        this_hyp_vectors = np.array(hyp_vectors[sent_id])
        this_ref_vectors = np.array(ref_vectors[sent_id])
        this_hyp_edits = hyp_edits[sent_id]
        this_ref_edits = ref_edits[sent_id]
        a = hyp_mass[sent_id]
        b = ref_mass[sent_id]
        if len(this_hyp_vectors) == 0 or len(this_ref_vectors) == 0:
            continue
        cost_matrix = ot.dist(
            this_hyp_vectors,
            this_ref_vectors,
            metric=uot_errant.config.cost
        )
        T = uot_errant.transport_plan(
            a=a, b=b, cost_matrix=cost_matrix
        )
        xtickslabels = [f"[{e.o_start}, {e.o_end}, {e.c_str}]" for mass, e in zip(b, this_ref_edits)]
        ytickslabels = [f"[{e.o_start}, {e.o_end}, {e.c_str}]" for mass, e in zip(a, this_hyp_edits)]
        fig1 = sns.heatmap(
            T,
            xticklabels=xtickslabels,
            yticklabels=ytickslabels,
            cmap='Blues',
            annot=True,
            fmt='.2f',
            annot_kws={"weight": "bold"}
        )
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45, va='top')
        # i += 1
        st.header(f'Sentence {sent_id}')
        st.write(f'SRC: {s}')
        st.write(f'HYP: {h}')
        st.write(f'REF: {r}')
        st.write(f'{a=}')
        st.write(f'{b=}')
        if fig1 == None:
            continue
        st.pyplot(fig1.get_figure())
        plt.close()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', default='meta_eval_data/SEEDA/outputs/all/GPT-3.5.txt')
    # parser.add_argument('--hyp', default='meta_eval_data/SEEDA/outputs/all/TemplateGEC.txt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)