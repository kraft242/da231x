# Evaluate

Follow the headings below to setup and compute the scores.

The instructions are adapted from the MultiGEC-2025 Data Providers repository.

## Virtual Environment

Create and activate a virtual environment, which can be either `conda` or `virtualenv`.

Execute the commands under the respective sub-heading to setup the environment.

### Conda

```bash
conda create -n evaluation_env python=3.12.5
conda activate evaluation_env
```

### Virtualenv

```bash
python3 -m venv evaluation_env
source evaluation_env/bin/activate
pip install -U pip setuptools wheel
```

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

## Install ipykernel

To be able to run the notebook in the created environment, you will need to install `ipykernel` by executing the below command:

```bash
pip install ipykernel
```

## Evaluate

You can now run the notebook `automatic_evaluation.ipynb` to perform the automatic evaluation.

The evaluation results are saved to scores.csv.

Do not forget to select the `evaluation_env` kernel in the notebook.
