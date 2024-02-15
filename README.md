# Setup
Install the conda environment.
```console
conda env create -f environment.yml
conda activate erllm
# avoid cuda coming with sentence-transformer, pytorch is already installed by environment.yml
(erllm) pip install --no-deps sentence-transformers==2.2.2
(erllm) python erllm_setup.py
```

The datasets are under `data/benchmark_datasets/existingDatasets`.

# Package Overview

| Module | Purpose |
| --- | --- |
| [erllm](#package-erllm) | Root package. Contains installation, documentation generation and helper code. |
| [erllm.calibration](#package-erllmcalibration) | Calibration analysis on entity matching LLM predictions. |
| [erllm.dataset](#package-erllmdataset) | Covers entity representation, dataset loading, downsampling.  The "DBpedia" submodule handles loading raw DBpedia data into subsampled CSV files, reading the raw data into SQLite database, interacting with it and generating labeled  datasets of matching and non-matching DBpedia entity pairs for benchmarking. |
| [erllm.dataset.dbpedia](#package-erllmdatasetdbpedia) | Handles DBPedia data including loading raw data into SQLite,  interaction, and generation of labeled datasets using token blocking for benchmarking. |
| [erllm.dataset.ditto](#package-erllmdatasetditto) | Convert existing datasets to DITTO format. |
| [erllm.discarder](#package-erllmdiscarder) | Explores the similarity-based discarder in isolation.  Computes and saves set-based and embedding-based similarities for pairs of entities,  Includes functionality to save results and computation time into similarity files, compute various discarder statistics, and generate visualizations. |
| [erllm.discarding_matcher](#package-erllmdiscarding_matcher) | Simulates and evaluates the similarity-based discarding matcher. Contains generation of performance plots, and analysis of time/performance trade-off. |
| [erllm.discarding_selective_matcher](#package-erllmdiscarding_selective_matcher) | Implements the discarding selective matcher. It includes functionalities for assessing classification performance, generating comparison tables and creating contour plots. |
| [erllm.ditto](#package-erllmditto) | Support for configuring DITTO to run on the DITTO datasets and subsequent evaluation and comparison to selective matcher. |
| [erllm.llm_matcher](#package-erllmllm_matcher) | Contains code to create prompts from datasets and get responses via OpenAI's API. These are saved into run files which serve as cache for all composite matchers. Also contains code to run and evaluate the LLM matcher. |
| [erllm.selective_classifier](#package-erllmselective_classifier) | Supports running selective classification on various datasets, evaluating the performance over ranges of threshold/coverage parameters, and generating tables and plots to visualize the classification performance. |
| [erllm.selective_matcher](#package-erllmselective_matcher) | Implements and evaluates the selective matcher and random labeling.  Supports running both methods across parameter ranges and datasets and generating comparison tables. |
| [erllm.serialization_cmp](#package-erllmserialization_cmp) | Compares entity serialization schemes, evaluating their performance with and without attribute names.  Also evaluates the impact of data errors. |

## Package: erllm

| Module | Purpose |
| --- | --- |
| [erllm_setup.py](erllm/erllm_setup.py) | Add .pth file to the site-packages directory of the current Python interpreter to make erllm discoverable. |
| [gen_docs.py](erllm/gen_docs.py) | Generate a package overview table and a table for each package's subfiles in a markdown file. |
| [utils.py](erllm/utils.py) | Utility functions for various tasks including file operations, mathematical calculations, and data manipulation. |

## Package: erllm.calibration

| Module | Purpose |
| --- | --- |
| [calibration_plots.py](erllm/calibration/calibration_plots.py) | Performs calibration analysis on language model predictions for different datasets. Calculating Brier Score and Expected Calibration Error (ECE). |
| [confidence_hist.py](erllm/calibration/confidence_hist.py) | Generate histograms of confidence scores per outcome (TP, TN, FP, FN). |
| [reliability_diagrams.py](erllm/calibration/reliability_diagrams.py) | Third party code from https://github.com/hollance/reliability-diagrams with some small changes. Calibration computation and visualization using reliability diagrams. |

## Package: erllm.dataset

| Module | Purpose |
| --- | --- |
| [entity.py](erllm/dataset/entity.py) | Contains Entity and OrderedEntity classes to represent entities and serialize them into strings for use in prompts. |
| [load_ds.py](erllm/dataset/load_ds.py) | Provides functions for loading benchmark data from CSV files into pandas DataFrames or lists of tuples representing entity pairs. |
| [sample_ds.py](erllm/dataset/sample_ds.py) | Provides a function for sampling elements from a dataset while preserving the label ratio. |
| [stats_ds.py](erllm/dataset/stats_ds.py) | This module provides functions to compute dataset statistics like the number of pairs. |

## Package: erllm.dataset.dbpedia

| Module | Purpose |
| --- | --- |
| [access_dbpedia.py](erllm/dataset/dbpedia/access_dbpedia.py) | Access the DBPedia SQLite database after it has been created by load_dbpedia.py. |
| [load_dbpedia.py](erllm/dataset/dbpedia/load_dbpedia.py) | Loads data from .txt file and loads it into SQLite tables.  The primary tables store DBpedia entities with key-value pairs, and an additional table stores matching pairs. |
| [sample_dbpedia.py](erllm/dataset/dbpedia/sample_dbpedia.py) | Provides functions for generating a sample dataset of entity pairs from the DBPedia database. The dataset includes both matching and non-matching pairs of entities. The matching pairs are generated based on known matches, non-matching pairs are generated by token blocking on random entities. |
| [token_blocking.py](erllm/dataset/dbpedia/token_blocking.py) | Provides functions for token blocking and clean token blocking in entity resolution tasks. |

## Package: erllm.dataset.ditto

| Module | Purpose |
| --- | --- |
| [to_ditto.py](erllm/dataset/ditto/to_ditto.py) | Provides functions for converting labeled pairs of entities to Ditto format and split them into train, validation, and test sets. |
| [to_ditto_runner.py](erllm/dataset/ditto/to_ditto_runner.py) | Generates Ditto datasets from existing datasets. |

## Package: erllm.discarder

| Module | Purpose |
| --- | --- |
| [discarder.py](erllm/discarder/discarder.py) | This module provides functions for computing set-based and embedding-based similarities for pairs of entities within a given dataset.  The set-based similarities include Jaccard, Overlap, Monge-Elkan, and Generalized Jaccard,  while embedding-based similarities use cosine and Euclidean distance metrics. Saves the results and computation time into similarity files which serve as cache for composite matchers including a discarder. |
| [discarder_eval.py](erllm/discarder/discarder_eval.py) | Computes various functions from similarity files, such as the number of false negatives as function of the number of discarded pairs. |
| [discarder_vis.py](erllm/discarder/discarder_vis.py) | Generates plots to visualize evaluation discarder statistitcs.  It includes functions to plot specific relations for a given dataset and generate combined plots for multiple datasets,  offering insights into various metrics such as false negatives, risk, false negative rate, and coverage. |

## Package: erllm.discarding_matcher

| Module | Purpose |
| --- | --- |
| [discarding_matcher.py](erllm/discarding_matcher/discarding_matcher.py) | This module provides functions for evaluating the performance of a discarding matcher utilizing run and similarity files. It calculates classification, cost and duration metrics. |
| [discarding_matcher_duration_cmp.py](erllm/discarding_matcher/discarding_matcher_duration_cmp.py) | Calculates speedup factor of discarding matcher over LLM matcher. |
| [discarding_matcher_runner.py](erllm/discarding_matcher/discarding_matcher_runner.py) | Runs the discarding matcher algorithm on multiple datasets with different threshold values. It calculates various performance metrics such as accuracy, precision, recall, F1 score, cost, and duration. |
| [discarding_matcher_tradeoff.py](erllm/discarding_matcher/discarding_matcher_tradeoff.py) | Generate and analyze performance/cost trade-off for the discarding matcher based on F1 decrease thresholds. Calculates F1 decrease, relative cost, and relative duration for each dataset and threshold. |
| [discarding_matcher_vis.py](erllm/discarding_matcher/discarding_matcher_vis.py) | Generates performance comparison plots for the discarding matcher. |

## Package: erllm.discarding_selective_matcher

| Module | Purpose |
| --- | --- |
| [discarding_selective_matcher.py](erllm/discarding_selective_matcher/discarding_selective_matcher.py) | Implements the discarding selective matcher and includes functions for evaluating its classification performance, cost and duration. |
| [discarding_selective_matcher_allstats_table.py](erllm/discarding_selective_matcher/discarding_selective_matcher_allstats_table.py) | Creates a table for comparing different matcher architectures based on their discarding error, cost, time and classification metrics. |
| [discarding_selective_matcher_contour.py](erllm/discarding_selective_matcher/discarding_selective_matcher_contour.py) | Create contour plots which map the discard and label fractions to the mean F1, precision, recall. |
| [discarding_selective_matcher_eval.py](erllm/discarding_selective_matcher/discarding_selective_matcher_eval.py) | Calculates the mean values across datasets for specified metrics,  based on the results obtained by running the discarding selective matcher. |
| [discarding_selective_matcher_metric_table.py](erllm/discarding_selective_matcher/discarding_selective_matcher_metric_table.py) | Create a table which shows one metric like mean F1 across different label and discard fractions. |
| [discarding_selective_matcher_runner.py](erllm/discarding_selective_matcher/discarding_selective_matcher_runner.py) | Runs and evaluates the discarding selective matcher for various configurations. |

## Package: erllm.ditto

| Module | Purpose |
| --- | --- |
| [add_to_ditto_configs.py](erllm/ditto/add_to_ditto_configs.py) | Copy the datasets in DITTO format to the ditto folder into the subfolder data/erllm. Add the new datasets to the configs.json file in the ditto folder. |
| [ditto_combine_predictions.py](erllm/ditto/ditto_combine_predictions.py) | Based on the stats of train and valid set and the results of running ditto, calculate the precision, recall, and F1 score for the total dataset. |
| [sm_ditto_comparison.py](erllm/ditto/sm_ditto_comparison.py) | Creates a table containing F1 scores for DITTO and SM across all datasets. |

## Package: erllm.llm_matcher

| Module | Purpose |
| --- | --- |
| [cost.py](erllm/llm_matcher/cost.py) | Provides cost calculations for language models based on specified configurations,  including input and output costs. |
| [evalrun.py](erllm/llm_matcher/evalrun.py) | Methods for reading run files, deriving classification decisions, and calculating classification and calibration metrics. |
| [gpt.py](erllm/llm_matcher/gpt.py) | Module for obtaining completions from the older OpenAI Completions API. |
| [gpt_chat.py](erllm/llm_matcher/gpt_chat.py) | Module for obtaining completions from the newer OpenAI Chat Completions API. |
| [llm_matcher.py](erllm/llm_matcher/llm_matcher.py) | Provides functions to evaluate the performance of the LLM matcher on a set of run files obtained from OpenAI's API.  It calculates various classification metrics, entropies, and calibration results. |
| [prompt_data.py](erllm/llm_matcher/prompt_data.py) | Handles serialization of labeled entity pairs and saves result into JSON. |
| [prompts.py](erllm/llm_matcher/prompts.py) | Combines serialized entities from JSON file with prompt prefix/postfix to create full prompts passed to OpenAI's API. |

## Package: erllm.selective_classifier

| Module | Purpose |
| --- | --- |
| [selective_classifier.py](erllm/selective_classifier/selective_classifier.py) | Run and evaluate selective classification. |
| [selective_classifier_runner.py](erllm/selective_classifier/selective_classifier_runner.py) | Runs selective classification over ranges of threshold/coverage parameters on multiple datasets. |
| [selective_classifier_tab.py](erllm/selective_classifier/selective_classifier_tab.py) | Create table of F1 scores per dataset for different coverages. |
| [selective_classifier_vis.py](erllm/selective_classifier/selective_classifier_vis.py) | Generates classification performance comparison plots for selective classification. |

## Package: erllm.selective_matcher

| Module | Purpose |
| --- | --- |
| [random_table.py](erllm/selective_matcher/random_table.py) | Create a latex comparison table of F1 scores between LLM matcher and random labeling at different label fractions. |
| [random_table_with_sd.py](erllm/selective_matcher/random_table_with_sd.py) | Creates a latex table displaying std. dev. of F1 scores for different fractions of random labeling. |
| [selective_matcher.py](erllm/selective_matcher/selective_matcher.py) | Implements the selective matcher and the labeling of randomly chosen predictions. It applies these to predictions on different datasets and calculates various classification metrics. |
| [selective_matcher_runner.py](erllm/selective_matcher/selective_matcher_runner.py) | This script runs and evaluates the selective matcher across parameter ranges and datasets. |
| [selective_matcher_vs_base_table.py](erllm/selective_matcher/selective_matcher_vs_base_table.py) | Create a latex comparison table of F1 scores between LLM matcher and selective matcher at different label fractions. |

## Package: erllm.serialization_cmp

| Module | Purpose |
| --- | --- |
| [attribute_comparison.py](erllm/serialization_cmp/attribute_comparison.py) | Creates per dataset and mean comparison tables for comparing entitiy serialization schemes with and without attributes names. |
| [data_errors.py](erllm/serialization_cmp/data_errors.py) | Generate comparison table of LLM matcher's mean F1, precision and recall across datasets in presence of data errors. |

# Datasets and their Format
The datasets are available in the repo but we describe how they can be downloaded and put in the right place for completeness.

## Dataset Format

Each dataset consists of five CSV files: `tableA.csv`, `tableB.csv`, `test.csv`, `train.csv`, and `valid.csv`.

### Example for Beer Dataset

`tableA.csv` and `tableB.csv` contain the entity descriptions in full. For example, the beer dataset is formatted as follows:

```csv
id,Beer_Name,Brew_Factory_Name,Style,ABV
12,Lagunitas Lucky 13 Mondo Large Red Ale,Lagunitas Brewing Company,American Amber / Red Ale,8.65%
13,Ruedrich's Red Seal Ale,North Coast Brewing Co.,American Amber / Red Ale,5.40%
14,Boont Amber Ale,Anderson Valley Brewing Company,American Amber / Red Ale,5.80%
15,American Amber Ale,Rogue Ales,American Amber / Red Ale,5.30%
``````

`test.csv`, `train.csv`, and `valid.csv` contain labeled pairs and repeat the entity descriptions. For example:

```csv
_id,label,table1.id,table2.id,table1.Beer_Name,table2.Beer_Name,table1.Brew_Factory_Name,table2.Brew_Factory_Name,table1.Style,table2.Style,table1.ABV,table2.ABV
0,0,1219,2470,Bulleit Bourbon Barrel Aged G'Knight,Figure Eight Bourbon Barrel Aged Jumbo Love,Oskar Blues Grill & Brew,Figure Eight Brewing,American Amber / Red Ale,Barley Wine,8.70%,-
1,0,492,1635,Double Dragon Imperial Red Ale,Scuttlebutt Mateo Loco Imperial Red Ale,Phillips Brewing Company,Scuttlebutt Brewing Co.,American Amber / Red Ale,American Strong Ale,8.20%,7.10%
2,1,3917,2224,Honey Basil Amber,Rude Hippo Honey Basil Amber,Rude Hippo Brewing Company,18th Street Brewery,American Amber / Red Ale,Amber Ale,7.40%,7.40%
```

We use unsupervised approaches and thus combine the pairs in `test.csv`, `train.csv`, and `valid.csv`.

## Download Datasets
The datasets are already prepared under `data/benchmark_datasets/existingDatasets`.  
For completeness we outline how to download the datasets (all except DBPedia) published by by Papadakis, George, Nishadi Kirielle, Peter Christen, and Themis Palpanas.  
If you want to work with the full DBPedia dataset directly follow the instructions under [DBPedia](#raw-dbpedia)

### Datasets except Raw DBPedia

1. **Download the Archive:**
   Download the `magellanExistingDatasets.tar.gz` file from https://zenodo.org/records/8164151.

2. **Navigate to the Directory:**
   Open a terminal or command prompt and navigate to the location where the downloaded file is stored.

3. **Extract the Archive:**
   Use the following command to extract the contents of the archive:
   ```bash
   tar -xvzf magellanExistingDatasets.tar.gz

4. **Move the Directory:**
    After extraction, move the existingDatasets directory to the desired location (data/benchmark_datasets in this case):
    ```bash
    mv existingDatasets data/benchmark_datasets/
    ```

5. **Verify the Structure:**
    Confirm that the directory structure now looks like this:
    ```text
    ├── data
    │   └── benchmark_datasets
    │       └── existingDatasets
    │           ├── ... (contents of the existingDatasets directory)
    ```

### Raw DBPedia

1. **Download the Archive:**
   Download the archive `dbpediaText.tar.gz` from https://zenodo.org/records/10059096.
2. **Extract and move to correct directory**
   Extract the archive and move the files `cleanDBPedia1out`, `cleanDBPedia2out`, `newDBPediaMatchesout`
   to data/dbpedia_raw.
3. **Verify the Structure:**
    Confirm that the directory structure now looks like this:
    ```text
    ├── data
    │   └── dbpedia_raw
    │       └── cleanDBPedia1out
    │       └── cleanDBPedia2out
    │       └── newDBPediaMatchesout
    ```

The python file in erllm/dataset/dbpedia are used to create the sample DBPedia dataset used in the work.

## Datasets except DBPedia
We evaluate on a wide range of datasets.
With the exception of DBpedia, we use them in the format published by
Papadakis, George, Nishadi Kirielle, Peter Christen, and Themis Palpanas. “A Critical Re-Evaluation of Benchmark Datasets for (Deep) Learning-Based Matching Algorithms.” arXiv, July 3, 2023. https://doi.org/10.48550/arXiv.2307.01231.

## DBPedia Raw Description
The [JedAIToolkit](https://github.com/scify/JedAIToolkit) contains the original copy of the DBPedia dataset in .jso format, which is a serialized Java object.  
To make this dataset easier to use, we submitted a [pull request](https://github.com/scify/JedAIToolkit/pull/66) to convert it to .txt files.  
We shared these with the authors of [JedAIToolkit](https://github.com/scify/JedAIToolkit) who uploaded it to https://zenodo.org/records/10059096.  
We do not use .csv because there is no fixed schema.

The files `cleanDBPedia1out`, `cleanDBPedia2out` contain the entities.
Each line corresponds to a different entity profile and has the following structure (where n is number of attributes, aname and aval are the attribute names and values):  
`numerical_id , uri , n ,  aname_0 , aval_0 , aname_1 , aval_1 ,...aname_n , aval_n`

That is, the separator is `space,space`.
`,` in the original data have been replaced with `,,`.
This must be accounted for when reading the data.

The file `newDBPediaMatchesout` contains matching profile pairs.
Each line has the format:  
`numerical_id_0 , numerical_id_1`
