# uot-errant
Code for the paper: "Grammatical Error Correction Evaluation by Optimally Transporting Edit Representation".  
This paper has been accepted to [Transactions of the Association for Computational Linguistics](https://transacl.org/index.php/tacl/index) (TACL). Currently, our paper is available in arXiv:
```bib
@misc{goto2026grammaticalerrorcorrectionevaluation,
      title={Grammatical Error Correction Evaluation by Optimally Transporting Edit Representation}, 
      author={Takumi Goto and Yusuke Sakai and Taro Watanabe},
      year={2026},
      eprint={2602.05419},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.05419}, 
}
```

This repository aims to provide experimental code for reproduction of our paper.
<!-- If you only want to use UOT-ERRANT, we recommend to use [gec-metrics](https://github.com/gotutiyan/gec-metrics). -->

# Install
```
pip install git+https://github.com/gotutiyan/uot-errant
python -m spacy download en_core_web_sm
```

# Usage

The interface follows [gec-metrics](https://github.com/gotutiyan/gec-metrics) [[Goto+ 25]](https://aclanthology.org/2025.acl-demo.50/).  
You can use `score_corpus()` for a corpus-level evaluation, and `score_sentence()` for a sentence-level evaluation.

```python
from uot_errant import UOTERRANT
config = UOTERRANT.Config(
    model='google/electra-base-discriminator',
    cost='euclidean',
    mass='norm',
    vectorize='ref-remove'
)
metric = UOTERRANT(config)
# metric.score_corpus()
# metric.score_sentence()
```

# Experimental code

Experimental scripts are available from [experiments/](./experiments/).
