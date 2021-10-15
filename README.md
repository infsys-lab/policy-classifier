# policy-classifier

This repository documents a [Random Forests](https://en.wikipedia.org/wiki/Random_forest) privacy-policy classifier based on input TF-IDF text features. This model expects input text to be in `markdown` format and has its own pre-processing pipelines. 

## Dependencies :mag:

This repository's code was tested with Python version `3.8.12`. To sync dependencies, we recommend creating a virtual environment and installing the relevant packages via `pip`:

```
$ pip install -r requirements.txt
```

## Initialization :fire:

1. To download manually annotated policies data from [Princeton](https://privacypolicies.cs.princeton.edu/data-release/data/) and prepare the data for training, simply execute:

    ```
    $ bash scripts/prepare_data.sh
    ```

3. **Optional:** Initialize git hooks to manage development workflows such as linting shell scripts and keeping python dependencies up-to-date:

    ```
    $ bash scripts/prepare_git_hooks.sh
    ```

## Usage :snowflake:

### Train

```
usage: train.py [-h] [--cv-splits <int>] [--debug]
                [--logging-level {debug,info,warning,error,critical}]
                [--n-jobs <int>] [--policies-csv <file_path>]
                [--precision-threshold <float>] [--random-seed <int>]
                [--scoring <str>]

optional arguments:
  --cv-splits            <int>
                         number of cross-validation splits (default: 5)
  --debug                <flag>
                         flag to debug script (default: False)
  --logging-level        {debug,info,warning,error,critical}
                         set logging level (default: info)
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
  -h, --help             <flag>
                         show this help message and exit
```

In order to train, cross-validate and evaluate the model, simply execute:

```
$ python3 src/train.py
```

This workflow will create a run directory in `./runs` and will dump all necessary logs, metrics and the final model checkpoint as a `dill.gz` compressed pickle. The dumped model checkpoint is a pipeline containing the `TfidfVectorizer` and `RandomForestClassifier` classes.

**Note:** This repository comes shipped with a ready-to-use model run in `./runs/run_1634294167`. Below we provide instructions on how to import and utilize the best model for downstream predictions.

### Import

In order to use a dumped model for downstream tasks, it is necessary to set up a virtual environment with the same Python and Scikit-Learn versions as this repository. Not doing so could result in unforeseen errors during the unpickling phase. Below is a code-snippet documenting how to import the best saved model:

```python
# load necessary dependencies
from gzip import decompress
from dill import loads

# load the raw compressed model as bytes
with open("path/to/model.dill.gz", "rb") as input_file_stream:
    model = input_file_stream.read()

# decompress and unpickle
model = loads(decompress(model))

# predict and provide probabilities for text being a privacy policy
model.predict_proba(["some markdown text", "some policy text"])[:,1]
```

## Test :microscope:

To run a `mypy` type-integrity test, execute:

```
$ mypy
```
