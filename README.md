# policy-classifier

This repository aims to reproduce a [Random Forests](https://en.wikipedia.org/wiki/Random_forest) privacy-policy classifier which was originally described in the [Princeton-Leuven Longitudinal Corpus of Privacy Policies](https://privacypolicies.cs.princeton.edu) paper. We train our classifier on TF-IDF text features from annotated privacy-policy markdown documents released alongside the aforementioned publication.

Our classifier was trained on a randomly split training set using a hyperparameter grid-search and 5-fold cross validation. It achieved a mean test ROC-AUC score of `0.955` during cross-validation. We then conducted post-hoc threshold tuning on a holdout set and found that a threshold of `0.752` was required to reach a precision of `0.99` for the privacy-policy or positive label. Finally after threshold tuning, we trained our classifier with the best performing hyperparameters on all training data to produce a production-ready model.

## Dependencies :mag:

1. This repository's code was tested with Python version `3.8.12`. To sync dependencies, we recommend creating a virtual environment and installing the relevant packages via `pip`:

    ```
    $ pip install -r requirements.txt
    ```

2. **Optional:** To access pre-computed data, this repository requires a working installation of Git [`LFS`](https://git-lfs.github.com/). We utilized version `v3.0.1` in our implementation.

## Initialization :fire:

1. To download and prepare annotated privacy-policies data for training, simply execute:

    ```
    $ bash scripts/prepare_data.sh
    ```

2. **Optional:** To clone the `policy-classifier-data` submodule containing a pre-computed final model (Git LFS object), simply execute:

    ```
    $ bash scripts/prepare_submodules.sh
    ```

3. **Optional:** Initialize git hooks to manage development workflows such as linting shell scripts and keeping python dependencies up-to-date:

    ```
    $ bash scripts/prepare_git_hooks.sh
    ```

## Usage :snowflake:

<details><summary>Train</summary><p>

```
usage: train.py [-h] [--cv-splits <int>] [--debug]
                [--logging-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                [--n-jobs <int>] [--policies-csv <file_path>]
                [--precision-threshold <float>] [--random-seed <int>]
                [--scoring <str>]

optional arguments:
  --cv-splits            <int>
                         number of cross-validation splits (default: 5)
  --debug                flag to debug script (default: False)
  --logging-level        {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                         set logging level (default: INFO)
  --n-jobs               <int>
                         number of parallel jobs, specify -1 to use all processors
                         (default: 1)
  --policies-csv         <file_path>
                         path to gold policies csv file (default:
                         data/1301_dataset.csv)
  --precision-threshold  <float>
                         precision threshold to match (default: 0.99)
  --random-seed          <int>
                         global random seed for RNGs (default: 42)
  --scoring              <str>
                         scoring metric for GridSearchCV (default: roc_auc)
  -h, --help             show this help message and exit
```

In order to train, cross-validate and evaluate the model, simply execute:

```
$ python3 src/train.py
```

This workflow will create a run directory in `./runs` and will dump all necessary logs, metrics and the final model checkpoint as a `dill` pickle. The dumped model checkpoint is a `sklearn` pipeline containing the `TfidfVectorizer` and `RandomForestClassifier` classes.

</p></details>
<details><summary>Predict</summary><p>

In order to use a dumped model for downstream tasks, it is necessary to set up a virtual environment with the same Python and Scikit-Learn versions as this repository. Not doing so could result in unforeseen errors during the unpickling phase. Below is a code-snippet documenting how to import and use the best saved model for prediction:

```python
# load necessary dependencies
from dill import load

# load the raw compressed model as bytes
with open("path/to/model.dill", "rb") as input_file_stream:
    model = load(input_file_stream)

# predict and provide probabilities for text being a privacy policy
model.predict_proba(["some markdown text", "some policy text"])[:,1]
```

</p></details>

## Test :microscope:

1. To run unit and integration tests, execute:

    ```
    $ pytest
    ```

2. To run a `mypy` type-integrity test, execute:

    ```
    $ mypy
    ```

## Citation :book:

If you found this repository helpful, we kindly ask you to cite our publication titled [Privacy and Customer’s Education: NLP for Information Resources Suggestions and Expert Finder Systems](https://link.springer.com/chapter/10.1007/978-3-031-05563-8_5): 

```bibtex
@InProceedings{10.1007/978-3-031-05563-8_5,
  author =       "Mazzola, Luca and Waldis, Andreas and Shankar, Atreya and
                  Argyris, Diamantis and Denzler, Alexander and Van Roey,
                  Michiel",
  editor =       "Moallem, Abbas",
  title =        "Privacy and Customer's Education: NLP for Information
                  Resources Suggestions and Expert Finder Systems",
  booktitle =    "HCI for Cybersecurity, Privacy and Trust",
  year =         "2022",
  publisher =    "Springer International Publishing",
  address =      "Cham",
  pages =        "62--77",
  isbn =         "978-3-031-05563-8"
}
```
