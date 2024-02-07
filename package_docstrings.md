
| Module | Purpose |
| --- | --- |
| erllm | Contains helpers and installer. |
| erllm.calibration | Calibration analysis on LLM predictions. |
| erllm.dataset | Covers entity representation, dataset loading, downsampling.  The "DBpedia" submodule handles loading raw DBpedia data into subsampled CSV files, reading the raw data into SQLite database, interacting with it and generating labeled  datasets of matching and non-matching DBpedia entity pairs for benchmarking. |
| erllm.dataset.dbpedia | Module for loading raw DBPedia data into subsampled csv files |
| erllm.dataset.ditto | Module for loading raw DBPedia data into subsampled csv files |
| erllm.discarder | Explores the similarity-based discarder in isolation.  Computes and saves set-based and embedding-based similarities for pairs of entities,  evaluates the impact of discarding based on various similarity functions, and visualizes the results. |
| erllm.discarding_matcher | Explores the similarity-based discarding matcher.  Simulates a discarding matcher, evaluates its performance on multiple datasets  with different threshold values, generates performance plots, and analyzes trade-off metrics based on F1 decrease thresholds. |
| erllm.discarding_selective_matcher | Explores the similarity-based discarding matcher.  Simulates a discarding matcher, evaluates its performance on multiple datasets  with different threshold values, generates performance plots, and analyzes trade-off metrics based on F1 decrease thresholds. |
| erllm.llm_matcher | Contains code to create prompts from datasets and get responses via OpenAI's API. |
| erllm.selective_classifier | Explores selective classification. |
| erllm.selective_matcher | Explores selective classification. |

| Module | Purpose |
| --- | --- |
| erllm.erllm_setup | Generates a .pth file with the absolute path to the parent folder of erllm and adds it to the site-packages directory of the current Python interpreter.  This makes erllm behave as if it were a installed third-party packages.  Yet it also supports code changes without reinstall. |
| erllm.gen_docs |  |
| erllm.utils |  |

| Module | Purpose |
| --- | --- |
| erllm.calibration.calibration_plots | Performs calibration analysis on language model predictions for different datasets, calculating Brier Score and Expected Calibration Error (ECE).  It generates visualizations of reliability diagrams and saves the calibration metrics in CSV files, organized by model configurations |
| erllm.calibration.confidence_hist |  |
| erllm.calibration.reliability_diagrams | Third party code from https://github.com/hollance/reliability-diagrams with some small changes. Code is licensed under the MIT license: MIT License  Copyright (c) 2020 M.I. Hollemans  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. |

| Module | Purpose |
| --- | --- |
| erllm.dataset.entity |  |
| erllm.dataset.load_ds |  |
| erllm.dataset.sample_ds |  |
| erllm.dataset.stats_ds |  |

| Module | Purpose |
| --- | --- |
| erllm.dataset.dbpedia.access_dbpedia | Access the DBPedia SQLite database after it has been created by load_dbpedia.py. |
| erllm.dataset.dbpedia.load_dbpedia | Reads raw data from specified paths and loads it into SQLite tables.  The primary tables store DBpedia entities with key-value pairs, and an additional table stores matching pairs.  Raw File Structure:  - The files `cleanDBPedia1out`, `cleanDBPedia2out` contain the entities. Each line corresponds to a different entity profile and has the following structure (where n is number of attributes, aname and aval are the attribute names and values):   `numerical_id , uri , n ,  aname_0 , aval_0 , aname_1 , aval_1 ,...aname_n , aval_n`  That is, the separator is `space,space`. `,` in the original data have been replaced with `,,`. This must be accounted for when reading the data.  - The file `newDBPediaMatchesout` contains matching profile pairs. Each line has the format:   `numerical_id_0 , numerical_id_1`  Table Structure: - Entities Table (e.g., dbpedia0, dbpedia1):   - Columns: id (INTEGER PRIMARY KEY), uri (TEXT), kv (JSON)  - Matches Table (e.g., dbpedia_matches):   - Columns: id0 (INTEGER), id1 (INTEGER) Note: The file is intended to be executed directly to populate the database. |
| erllm.dataset.dbpedia.sample_dbpedia |  |
| erllm.dataset.dbpedia.to_ditto |  |
| erllm.dataset.dbpedia.token_blocking |  |

| Module | Purpose |
| --- | --- |
| erllm.dataset.ditto.add_to_ditto_configs |  |
| erllm.dataset.ditto.to_ditto |  |
| erllm.dataset.ditto.to_ditto_runner |  |

| Module | Purpose |
| --- | --- |
| erllm.discarder.discarder |  |
| erllm.discarder.discarder_eval |  |
| erllm.discarder.discarder_vis |  |

| Module | Purpose |
| --- | --- |
| erllm.discarding_matcher.discarding_matcher |  |
| erllm.discarding_matcher.discarding_matcher_duration_cmp |  |
| erllm.discarding_matcher.discarding_matcher_runner | This script runs the discarding matcher algorithm on multiple datasets with different threshold values. It calculates various performance metrics such as accuracy, precision, recall, F1 score, cost, and duration. The results are stored in a pandas DataFrame and saved as a CSV file. |
| erllm.discarding_matcher.discarding_matcher_tradeoff | Generate and analyze performance trade-off metrics for the discarding matcher based on F1 decrease thresholds. Reads performance metrics from a CSV file, calculates F1 decrease, relative cost, and relative duration for each dataset and threshold. |
| erllm.discarding_matcher.discarding_matcher_vis | This script generates performance comparison plots for the discarding matcher. It reads performance metrics from a CSV file, filters the data based on selected metrics, and creates line plots for each dataset with different configurations, such as all metrics, no cost, and F1 with cost. It also creates plots with showing the performance on all datasets at once. |

| Module | Purpose |
| --- | --- |
| erllm.discarding_selective_matcher.discarding_selective_matcher |  |
| erllm.discarding_selective_matcher.discarding_selective_matcher_allstats_table |  |
| erllm.discarding_selective_matcher.discarding_selective_matcher_contour |  |
| erllm.discarding_selective_matcher.discarding_selective_matcher_eval |  |
| erllm.discarding_selective_matcher.discarding_selective_matcher_metric_table |  |
| erllm.discarding_selective_matcher.discarding_selective_matcher_runner |  |

| Module | Purpose |
| --- | --- |
| erllm.llm_matcher.cost |  |
| erllm.llm_matcher.evalrun | Methods for reading run files, deriving classification decisions, and calculating classification and calibration metrics |
| erllm.llm_matcher.gpt |  |
| erllm.llm_matcher.gpt_chat |  |
| erllm.llm_matcher.llm_matcher | Provides functions to evaluate the performance of the LLM Mathcer on a set of run files produced by the OpenAI GPT API.  It calculates various classification metrics, entropies, and calibration results.  The evaluation results are saved as JSON files for individual runs and aggregated into a CSV file for further analysis. |
| erllm.llm_matcher.prompt_data |  |
| erllm.llm_matcher.prompts |  |

| Module | Purpose |
| --- | --- |
| erllm.selective_classifier.selective_classifier |  |
| erllm.selective_classifier.selective_classifier_runner | This script runs the discarding matcher algorithm on multiple datasets with different threshold values. It calculates various performance metrics such as accuracy, precision, recall, F1 score, cost, and duration. The results are stored in a pandas DataFrame and saved as a CSV file. |
| erllm.selective_classifier.selective_classifier_tab | This script generates performance comparison plots for the discarding matcher. It reads performance metrics from a CSV file, filters the data based on selected metrics, and creates line plots for each dataset with different configurations, such as all metrics, no cost, and F1 with cost. It also creates plots with showing the performance on all datasets at once. |
| erllm.selective_classifier.selective_classifier_vis | This script generates performance comparison plots for the discarding matcher. It reads performance metrics from a CSV file, filters the data based on selected metrics, and creates line plots for each dataset with different configurations, such as all metrics, no cost, and F1 with cost. It also creates plots with showing the performance on all datasets at once. |

| Module | Purpose |
| --- | --- |
| erllm.selective_matcher.random_table | This script generates performance comparison plots for the discarding matcher. It reads performance metrics from a CSV file, filters the data based on selected metrics, and creates line plots for each dataset with different configurations, such as all metrics, no cost, and F1 with cost. It also creates plots with showing the performance on all datasets at once. |
| erllm.selective_matcher.random_table_with_sd | This script generates performance comparison plots for the discarding matcher. It reads performance metrics from a CSV file, filters the data based on selected metrics, and creates line plots for each dataset with different configurations, such as all metrics, no cost, and F1 with cost. It also creates plots with showing the performance on all datasets at once. |
| erllm.selective_matcher.selective_matcher | Defines functions to manually label predictions by selecting the k most uncertain,  k most uncertain negative, and k random predictions from a given set. It applies these labeling strategies to predictions on different datasets,  calculates various classification metrics, and saves the results for comparison. |
| erllm.selective_matcher.selective_matcher_runner | This script runs the discarding matcher algorithm on multiple datasets with different threshold values. It calculates various performance metrics such as accuracy, precision, recall, F1 score, cost, and duration. The results are stored in a pandas DataFrame and saved as a CSV file. |
| erllm.selective_matcher.selective_matcher_vs_base_table | This script generates performance comparison plots for the discarding matcher. It reads performance metrics from a CSV file, filters the data based on selected metrics, and creates line plots for each dataset with different configurations, such as all metrics, no cost, and F1 with cost. It also creates plots with showing the performance on all datasets at once. |
