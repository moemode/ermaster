# Setup
Install the conda environment.  
The datasets are under `data/benchmark_datasets/existingDatasets`.

# Structure

## Module Overview

| Module            | Purpose                      |
| --------------- | ------------------------- |
| [datasets](#module-datasets)    | Covers entity representation, dataset loading, downsampling. The "DBpedia" submodule handles loading raw DBpedia data into subsampled CSV files, reading the raw data into SQLite database, interacting with it and generating labeled datasets of matching and non-matching DBpedia entity pairs for benchmarking.|
| [llm_matcher](#module-llm_matcher) | Contains code to create prompts from datasets and get responses via OpenAI's API.|

## Module datasets
| Module            | Purpose                      |
| --------------- | ------------------------- |
| [dataset/entity.py](erllm/dataset/load_ds.py)   | Representation of entities involving initialization, generating string or token representations.
| [dataset/load_ds.py](erllm/dataset/load_ds.py)   | Code for reading datasets from csv into dataframes or list of profile pairs.|
| [dataset/sample_ds.py](erllm/dataset/sample_ds.py)   | Downsample datasets.|
| [dataset/stats_ds.py](erllm/dataset/stats_ds.py)   | Get number of instances, positive and negatives in datasets.|

### Submodule DBpedia
Module for loading raw DBPedia data into subsampled csv files
| Module            | Purpose                      |
| --------------- | ------------------------- |
| [dataset/dbpedia/load_dbpedia.py](erllm/dataset/dbpedia/load_dbpedia.py)   | Reads raw DBpedia data into SQLite database. |
[dataset/dbpedia/access_dbpedia.py](erllm/dataset/dbpedia/access_dbpedia.py)   | Database interaction functions for querying and retrieving DBpedia entities and matching pairs.
| [dataset/dbpedia/token_blocking.py](erllm/dataset/dbpedia/access_dbpedia.py)   | Implements dirty and clean-clean token blocking 
| [dataset/dbpedia/sample_dbpedia.py](erllm/dataset/dbpedia/sample_dbpedia.py)   | Generates a labeled dataset of matching and non-matching DBpedia entity pairs.|

## Module llm_matcher

Contains code to create prompts from datasets and get responses via OpenAI's Api.
| File            | Purpose                      |
| --------------- | ------------------------- |
| [llm_matcher/prompt_data.py](erllm/llm_matcher/prompt_data.py) | Serialize entities intro string represenation for use in prompt template
| [llm_matcher/prompts.py](erllm/llm_matcher/prompts.py) | Takes output of `prompt_data.py` and convert it to JSON files containing full prompts based on template choice|
| [llm_matcher/gpt.py](erllm/llm_matcher/gpt.py) | Takes prompts prepared by `prompts.py`and generates completions via OpenAI API. The results of runs are saved in `runfiles` which store the full API response and information about the profile pairs.|
| [llm_matcher/cost.py](erllm/llm_matcher/cost.py) | Given prompts, calculates the cost of generating completions specified models and datasets. |


# Datasets and their Format
The datasets are available in the repo but we describe how they can be downloaded and put in the right place for completion.

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
