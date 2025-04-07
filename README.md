# DA231X Degree Project in Computer Science and Engineering / Examensarbete i datalogi och datateknik

Work in the course DA231X Examensarbete i datalogi och datateknik, avancerad nivå 30,0 hp

## Requirements

### SweLL-gold Corpus

Make sure you have the SweLL-gold corpus locally downloaded and available.

Follow the steps below:

1. Create a directory `data/` in the the root of this repository.
2. Copy the entire `swedish/` directory from the SweLL-gold repository into `data/`
    - The directory should be structured as follows: `./data/swedish/SweLL_gold/`.

### Python Virtual Environment

Create a python virtual environment with the following command:

```bash
$ python -m venv ./venv
```

Activate the environment with:

```bash
$ source ./venv/bin/activate
```

After activation, install the dependencies with:

```bash
$ pip install -r requirements.txt
```

## Running

You will first need to parse the SweLL-gold corpus into a Hugging-Face compatible dataset.

This is done with the `create_datasets.ipynb` notebook.

Open the notebook and run all cells to create both the minimal-edit and fluency-edit datasets in the `datasets/` directory, which will be automatically placed in the repository root.

Then run the `train_minimal.ipynb` and `train_fluency.ipynb` notebooks to train the respective models.

The models are saved under the `models/` directory, which is also automatically created.
