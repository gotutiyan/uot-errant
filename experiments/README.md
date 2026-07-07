This directory contains experimental code.

# Preparation
```sh
# Download meta-evaluation datasets
gecmetrics-prepare-meta-eval

# Download CLEME
git clone https://github.com/THUKElab/CLEME.git

# Need spacy model to use ERRANT-like metrics
python -m spacy download en_core_web_sm

# Download pre-trained weights of SOME using `gdown` module.
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/1uoAReQK3f5g9CEy8rV4haSzXll8NqVHW/view
unzip gfm-models.zip
```

# Figure 2 (Average number of edits per sentence)
```shell
python gmeg.py --density 
```

The outputs will be saved to `exp-outputs/ExpGMEG/density.{png|pdf}`.


# Table 1 (Results)
For SEEDA,
```sh
python seeda.py
```
This scripts automatically download references, such as `10 Refs` and `e-fluency`.
The results will be saved to `exp-outputs/ExpSEEDA/*` and LaTeX strings will output to the terminal.

For GMEG-data,
```sh
python gmeg.py
```
This scripts automatically download GMEG-Data.
The results will be saved to `exp-outputs/ExpGMEG/*`.  
The LaTeX strings will output to the terminal.

(We manually concatenate SEEDA's and GMEG's results to make Table 1.)

# Figure 3 (Scatter plot of SEEDA systems)
```sh
python scatter.py
```

The outputs will be saved to `exp-outputs/ExpScatter/scatter.{png|pdf}`.

# Figure 4 (Visualization of edit alignment)
```sh
streamlit run app_case_study.py 
```

Then go to the URL shown in the terminal.

# Table 3 (Norm of edit vector)
```sh
python edit_vector_norm.py
```

The outputs will be saved to `exp-outputs/ExpNorm/*`.

The results with LaTeX format (without headers and footers, such as `\begin{table}`) will be output to the terminal, like:
```
ORTH       & 0.170 $_\text{± 0.406}$ \\
PUNCT      & 0.835 $_\text{± 0.597}$ \\
...
OTHER      & 1.318 $_\text{± 0.685}$ \\
SPELL      & 1.409 $_\text{± 0.610}$ \\
```

# Figure 5 (Visualization of edit vector)
```sh
python edit_vector_visualization.py
```

The outputs will be saved to `exp-outputs/ExpEditVectorVisualization/google/electra-base-discriminator-cat2-tsne.{png|pdf}`.

# Figure 6 (Detail analysis of sentence-level correlations)
```sh
python pairwise-analysis.py
```

The outputs will be saved to `exp-outputs/ExpPairwiseAnalysis/*`.
- `errant.{png|pdf}` is Figure 6 (a) in the paper.
- `PTERRANT-ERRANT.{png|pdf}` is Figure 6 (b) in the paper.
- `UOTERRANT-ERRANT.{png|pdf}` is Figure 6 (c) in the paper.

# Figure 7 (Efficiency)
```sh
python efficiency.py
```

The outputs will be saved to `exp-outputs/ExpEfficiency/efficiency.{png|pdf}`.