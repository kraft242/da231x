# Evaluate and Plot

This directory handles the evaluation and plotting.

Evaluation instructions can be found under the `Evaluate` heading.

Pre-computed scores are also available, which is explained under the `Plot` heading.

## Plot

The scores used in the report can be found in the `scores_long.csv` file, which can also be visualized and placed into tables with the `plot.ipynb` notebook.

Create a virtual enviroment for plotting and install the required packages before running the plot notebook:

```bash
python -m venv plot-env
source plot-env/bin/activate
pip install -r plot_requirements.txt
```

You can now run the `plot.ipynb` notebook, just make sure to select the `plot-env` kernel.

## Evaluate

Follow the headings below to setup and compute the scores.

The instructions are adapted from the MultiGEC-2025 Data Providers repository.

### Virtual Environment

You can use the same virtual environment as when training the models.

Make sure you have activated the environment before installing the packages, i.e run: `$ source ../.venv/bin/activate`.

### Install Packages

Execute the commands under each respective sub-heading to install the automatic-evaluation packages.

#### ERRANT

```bash
pip install regex pandas syntok

git clone https://github.com/cainesap/errant
cd errant
pip install -e .
cd ../

git clone https://github.com/cainesap/spacy_conll
cd spacy_conll
pip install -e .
cd ../

pip install spacy-udpipe
mkdir spacy_udpipe_models
python download_spacy_udpipe_models.py
mkdir spacy_models
python -m spacy download en_core_web_sm
python download_spacy_english_model.py
```

#### GLEU

```bash
git clone https://github.com/shotakoyama/gleu
cd gleu
pip install -e .
cd ../
```

#### Scribendi Score

```bash
git clone https://github.com/robertostling/scribendi_score
cd scribendi_score
pip install -r requirements.txt
cd ../
```

After this, you will also need to move or copy the `scribeni.py` module into this directory, i.e run `cp scribendi/scribendi.py .`

### Run Evaluation

You can now run the notebook `automatic_evaluation.ipynb` to perform the automatic evaluation.

The pairwise evaluation results are saved to `scores.csv` and the corpuswise results are saved to `corpus_scores.csv`.

Do not forget to select the `venv` kernel in the notebook.
