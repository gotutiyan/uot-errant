from gec_metrics.metrics import ERRANT
from gecommon import apply_edits
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import ot
from dataclasses import dataclass
from tqdm import tqdm
import torch.nn.functional as F
import math
import hashlib
import errant
import itertools

class UOTERRANT(ERRANT):
    @dataclass
    class Config(ERRANT.Config):
        '''UOT-ERRANT configuration
            - model (str): Embedding model to compute edit vectors
            - batch_size (int): Batch size during embedding
            - reg (float): The weight for the entoropy reguralization term for the objective function in UOT problem.
            - reg_m (float): The degree to which all quantities of transport are enforced.
            - cost (str): How to define cost.
                - "euclidean" uses Euclidean distance of edit vectors
                - "cosine" uses cosine similarity of edit vectors
            - mass (str): How to define mass.
                - "norm" uses L2 norm of edit vector
                - "uniform" uses uniform weights, i.e. all weights are 1/N when N edits exist.
            - vectorize (str): How to compute edit vector.
                - "ref-remove": Observes the impact when an edit is removed from a complete correction.
                - "src-add": Observes the impact when an edit is added to a errorneous sentence.
            - 
        '''
        model: str = 'google/electra-base-discriminator'
        batch_size: int = 32
        reg: float = 0.1
        reg_m: float = 0.1
        cost: str = 'euclidean'
        mass: str = 'norm'
        vectorize: str = 'ref-remove'

    class Score(ERRANT.Score):
        @property
        def precision(self) -> float:
            '''Calculate the precision.'''
            try:
                return self.tp / (self.tp + self.fp)
            except ZeroDivisionError:
                return 1.0
        
        @property
        def recall(self) -> float:
            '''Calculate the recall '''
            try:
                return self.tp / (self.tp + self.fn)
            except ZeroDivisionError:
                return 1.0

        @property
        def f(self) -> float:
            '''Calculate the F-beta score. '''
            p = self.precision
            r = self.recall
            beta = self.beta
            try:
                f = float((1+(beta**2))*p*r)/(((beta**2)*p)+r) if p+r else 0.0
            except ZeroDivisionError:
                return 1.0
            return f

    def __init__(self, config=None):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(self.config.model).eval()
        if torch.cuda.is_available():
            self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.cache_emb = dict()

    def __init__(self, config=None):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(self.config.model).eval()
        if torch.cuda.is_available():
            self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.cache_emb = dict()

    def load_cache(self, text):
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in self.cache_emb:
            return self.cache_emb[key].copy()
        else:
            return None
        
    def save_cache(self, text, vec):
        key = hashlib.sha256(text.encode()).hexdigest()
        self.cache_emb[key] = vec

    def mass_func(self, edit_vectors):
        '''Calculate mass for the input vectors.
        
        Args:
            edit_vectors (array-like): 
                Vector for each edit.
                The shape should be (num_samples, num_hidden_size).
        
        Returns:
            np.ndarray: The mass of the vectors. The shape is (num_samples).
        '''
        if not isinstance(edit_vectors, np.ndarray):
            edit_vectors = np.array(edit_vectors)
        if edit_vectors.shape[0] == 0:
            return []
        if self.config.mass == 'norm':
            # The mass of the vector is defined as its L2 norm.
            # Note: The total mass is different in each sentence.
            mass = np.linalg.norm(edit_vectors, axis=-1)
        if self.config.mass == 'uniform':
            # All mass is equal to 1/N.
            # Note: The total mass is always 1.
            mass = np.full(edit_vectors.shape[0], 1/edit_vectors.shape[0])
        assert mass.shape[0] == edit_vectors.shape[0]
        return mass

    def mean_pooling(
        self,
        states: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        '''Compute mean pooling. Only the representaion with mask==1 are used.
        
        Args:
            states (torch.Tensor): The token-level representation.
                The shape is (num_batch, sequence_length, hidden_size)
            mask: torch.Tensor: The mask indicates padding or not.
                The shape is (num_batch, sequence_length)
        
        Returns:
            torch.Tensor: The mean pooled representation.
                The shape is (num_batch, hidden_size)
        '''
        states[mask == 0] = 0  # batch x seq_len x hidden
        sum_logits = torch.sum(states, dim=1)  # batch x hidden
        length = torch.sum(mask, dim=-1)  # batch x
        pooled_logits = torch.div(sum_logits.transpose(1, 0), length).transpose(1, 0)  # batch x hidden
        return pooled_logits

    @torch.no_grad()
    def embed(
        self,
        sentences: list[str]
    ) -> np.ndarray:
        '''Embeds each sentence.
        
        Args:
            sentences (list[str]): Sentences to be embedded.
        
        Returns:
            np.ndarray: Embeddings. The shape is (num_samples, num_hidden_dim).
        '''
        num_sents = len(sentences)
        bsz = self.config.batch_size
        sent_vectors = None
        for i in range(0, num_sents, bsz):
            batch = sentences[i: i+bsz]
            encode = self.tokenizer(
                batch,
                max_length=128,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            encode = {k: v.to(self.model.device) for k, v in encode.items()}
            out = self.model(**encode)
            vec = self.mean_pooling(
                out.last_hidden_state,
                encode['attention_mask']
            )
            if sent_vectors is None:
                sent_vectors = vec.cpu().numpy()
            else:
                sent_vectors = np.vstack(
                    [sent_vectors, vec.cpu().numpy()]
                )
        return sent_vectors
        
    def edit_vector(
        self,
        sources: list[str],
        edits: list[list[errant.edit.Edit]]
    ) :
        '''Compute edit vectors.

        Args:
            sources (list[str]):
                Source sentences.
            edits (list[list[errant.edit.Edit]]):
                Edits. The shape is (num_sents, num_edits).
        
        Returns:
            list[list[list[float]]]:
                Edit vectors.
                The shape is (num_sents, num_edits, hidden_size)
        '''
        num_sents = len(sources)
        cor = [apply_edits(sources[i], edits[i]) for i in range(num_sents)]
        inputs = []
        sent_ids = []
        for sent_id in range(num_sents):
            this_src = sources[sent_id]
            this_edits = edits[sent_id]
            if self.config.vectorize == 'ref-remove':
                inputs.append(cor[sent_id])
            elif self.config.vectorize == 'src-add':
                inputs.append(sources[sent_id])
            sent_ids.append(sent_id)
            for edit_id in range(len(this_edits)):
                if self.config.vectorize == 'ref-remove':
                    edits_wo_i = [this_edits[j] for j in range(len(this_edits)) if edit_id != j]
                    cor_wo_i = apply_edits(this_src, edits_wo_i)
                    inputs.append(cor_wo_i)
                elif self.config.vectorize == 'src-add':
                    cor_i = apply_edits(this_src, [this_edits[edit_id]])
                    inputs.append(cor_i)
                sent_ids.append(sent_id)
        # Below, it loads pre-compute embeddings from the cache.
        # If the input is no cached, load_cache() returns None
        embeddings = [self.load_cache(inp) for inp in inputs]
        non_cached_ids = [i for i in range(len(inputs)) if embeddings[i] is None]
        if len(non_cached_ids) > 0:
            partial_embs = self.embed([inputs[i] for i in non_cached_ids])
            for i, j in enumerate(non_cached_ids):
                embeddings[j] = partial_embs[i]
                # self.save_cache(inputs[j], partial_embs[i])
        
        edit_vectors = [[] for _ in range(num_sents)]
        edit_orig_embs = [[] for _ in range(num_sents)]
        cor_vectors = [[] for _ in range(num_sents)]
        mass = [[] for _ in range(num_sents)]
        for i in range(len(embeddings)):
            if i == 0 or sent_ids[i - 1] != sent_ids[i]:
                base_emb = embeddings[i]
                cor_vectors[sent_ids[i]] = base_emb
            else:
                if self.config.vectorize == 'ref-remove':
                    edit_vectors[sent_ids[i]].append(base_emb - embeddings[i])
                elif self.config.vectorize == 'src-add':
                    edit_vectors[sent_ids[i]].append(embeddings[i] - base_emb)
                edit_orig_embs[sent_ids[i]].append(embeddings[i])
        for i in range(num_sents):
            mass[i] = self.mass_func(edit_vectors[i])
            assert len(edit_vectors[i]) == len(edits[i]) == len(mass[i]), f"{len(edit_vectors[i])=}, {len(edits[i])=}, {len(mass)=}"
        return edit_vectors, mass

    def transport_plan(
        self,
        a: np.ndarray,
        b: np.ndarray,
        cost_matrix: np.ndarray
    ) -> np.ndarray:
        '''Compute optimal transport plan.
        
        Args:
            - a (np.ndarray): Mass before transport.
            - b (np.ndarray): Mass after transport.
            - cost_matric (np.ndarray): Cost matrix. This is a two dimensional matrix and cost_matrix_{ij} means the difficulty of transporting from a_i to b_j.

        Returns:
            np.ndarray: The optimal transport plan. The shape is (|a|, |b|). The index [i][j] is a transported mass from a_i to b_j.
        '''
        return ot.unbalanced.sinkhorn_stabilized_unbalanced(
            a=a, b=b, M=cost_matrix,
            reg=self.config.reg,
            reg_m=self.config.reg_m,
        )
        

    def score_base(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[str]
    ) -> list[list[dict[str, "Score"]]]:
        '''Calculate scores while retaining sentence and reference boundaries.
            The results can be aggregated according to the purpose,
                e.g., at sentence-level or corpus-level.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            list[list[dict[str, "Score"]]]: The verbose scores.
                - The list shape is (num_sents, num_refs)
                - The dict contains error type-wise scores.
        '''
        num_sents = len(sources)
        num_refs = len(references)
        scores = []  # The shape will be: (num_sents, num_refs)
        hyp_edits = [
            self.edit_extraction(
                sources[sent_id],
                hypotheses[sent_id]
            ) for sent_id in range(num_sents)
        ]
        ref_edits_list = [
            [
                self.edit_extraction(
                    sources[sent_id],
                    references[ref_id][sent_id]
                ) for sent_id in range(num_sents)
            ] for ref_id in range(num_refs)
        ]
        hyp_vectors, hyp_mass = self.edit_vector(sources, hyp_edits)
        ref_vectors, ref_mass = [], []
        for ref_id in range(num_refs):
            vec, mass = self.edit_vector(
                sources,
                ref_edits_list[ref_id]
            )
            ref_vectors.append(vec)
            ref_mass.append(mass)

        for sent_id in range(num_sents):
            sent_scores = []  # The shape will be: (num_refs, )
            for ref_id in range(len(ref_vectors)):
                this_hyp_vectors = np.array(hyp_vectors[sent_id])
                this_ref_vectors = np.array(ref_vectors[ref_id][sent_id])
                this_score = dict()
                a = hyp_mass[sent_id]
                b = ref_mass[ref_id][sent_id]
                if len(this_hyp_vectors) > 0 and len(this_ref_vectors) > 0:
                    cost_matrix = ot.dist(
                        this_hyp_vectors,
                        this_ref_vectors,
                        metric=self.config.cost
                    )
                    T = self.transport_plan(a, b, cost_matrix) # shape: (num_hyp_edits, num_ref_edits)
                    transported_a = T.sum(axis=1)  # shape: (num_hyp_edits)
                    transported_b = T.sum(axis=0)  # shape: (num_ref_edits)
                else:
                    # When len(this_hyp_vectors) == 0 or len(this_ref_vectors) == 0,
                    #   no transportation will occurs.
                    transported_a = np.zeros_like(a)
                    transported_b = np.zeros_like(b)
                for edit_id, h_edit in enumerate(hyp_edits[sent_id]):
                    etype = h_edit.type
                    this_score[etype] = this_score.get(
                        etype, self.Score(beta=self.config.beta)
                    )
                    # TP := The transported amount
                    this_score[etype].tp += float(transported_a[edit_id])
                    # FP := The non-transported amount from a source
                    this_score[etype].fp += float(a[edit_id] - transported_a[edit_id])
                for edit_id, r_edit in enumerate(ref_edits_list[ref_id][sent_id]):
                    etype = r_edit.type
                    this_score[etype] = this_score.get(
                        etype, self.Score(beta=self.config.beta)
                    )
                    # FN := The non-transported amount to a reference
                    this_score[etype].fn += float(b[edit_id] - transported_b[edit_id])
                sent_scores.append(this_score)
            scores.append(sent_scores)
        return scores
