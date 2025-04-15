# Evaluate

Follow the headings below to setup and compute the scores.

The instructions are adapted from the MultiGEC-2025 Data Providers repository.

## Virtual Environment

You can use the same virtual environment as before.

Make sure you have activated the environment before installing the packages, i.e run: `$ source ../.venv/bin/activate`.

## Install Packages

Execute the commands under each respective sub-heading to install the automatic-evaluation packages.

### ERRANT

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

### GLEU:

```bash
git clone https://github.com/shotakoyama/gleu
cd gleu
pip install -e .
cd ../
```

### Scribendi Score

```bash
git clone https://github.com/robertostling/scribendi_score
cd scribendi_score
pip install -r requirements.txt
cd ../
```

After this, you will also need to move or copy the `scribeni.py` module into this directory, i.e run `cp scribendi/scribendi.py .`

## Evaluate

You can now run the notebook `automatic_evaluation.ipynb` to perform the automatic evaluation.

The pairwise evaluation results are saved to `scores.csv` and the corpuswise results are saved to `corpus_scores.csv`.

Do not forget to select the `venv` kernel in the notebook.
