# Files

| File            | Purpose                      |
| --------------- | ------------------------- |
| [sample_ds.py](sample_ds.py)    | Downsample datasets |

# Datasets and their Format
We evaluate on a wide range of datasets.
With the exception of DBpedia, we use them in the format published by
Papadakis, George, Nishadi Kirielle, Peter Christen, and Themis Palpanas. “A Critical Re-Evaluation of Benchmark Datasets for (Deep) Learning-Based Matching Algorithms.” arXiv, July 3, 2023. https://doi.org/10.48550/arXiv.2307.01231.

## Download

Their paper repo is at https://github.com/gpapadis/DLMatchers/.  
The datasets themselves are at https://zenodo.org/records/8164151.
Make sure to extract the `magellanExistingDatasets.tar.gz` so that the existingDatasets is under
data/benchmark_datasets/existingDatasets.

### Extracting `magellanExistingDatasets.tar.gz`

1. **Download the Archive:**
   Download the `magellanExistingDatasets.tar.gz` file from the source.

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
## Format

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

## Subsampling

| File            | Purpose                      |
| --------------- | ------------------------- |
| [sample_ds.py](sample_ds.py)    | Downsample |