# Package Overview

| Module | Purpose |
| --- | --- |
| [erllm](#package-erllm) | Contains helpers and installer. |
| [erllm.calibration](#package-erllmcalibration) | Calibration analysis on LLM predictions. |
| [erllm.dataset](#package-erllmdataset) | Covers entity representation, dataset loading, downsampling.  The "DBpedia" submodule handles loading raw DBpedia data into subsampled CSV files, reading the raw data into SQLite database, interacting with it and generating labeled  datasets of matching and non-matching DBpedia entity pairs for benchmarking. |
| [erllm.dataset.dbpedia](#package-erllmdatasetdbpedia) | Module for loading raw DBPedia data into subsampled csv files |
| [erllm.dataset.ditto](#package-erllmdatasetditto) | Module for loading raw DBPedia data into subsampled csv files |
| [erllm.discarder](#package-erllmdiscarder) | Explores the similarity-based discarder in isolation.  Computes and saves set-based and embedding-based similarities for pairs of entities,  evaluates the impact of discarding based on various similarity functions, and visualizes the results. |
| [erllm.discarding_matcher](#package-erllmdiscarding_matcher) | Explores the similarity-based discarding matcher.  Simulates a discarding matcher, evaluates its performance on multiple datasets  with different threshold values, generates performance plots, and analyzes trade-off metrics based on F1 decrease thresholds. |
| [erllm.discarding_selective_matcher](#package-erllmdiscarding_selective_matcher) | Explores the similarity-based discarding matcher.  Simulates a discarding matcher, evaluates its performance on multiple datasets  with different threshold values, generates performance plots, and analyzes trade-off metrics based on F1 decrease thresholds. |
| [erllm.llm_matcher](#package-erllmllm_matcher) | Contains code to create prompts from datasets and get responses via OpenAI's API. |
| [erllm.selective_classifier](#package-erllmselective_classifier) | Explores selective classification. |
| [erllm.selective_matcher](#package-erllmselective_matcher) | Explores selective classification. |

## Package: erllm

| Module | Purpose |
| --- | --- |
| [erllm_setup.py](erllm/erllm_setup.py) | Generates a .pth file with the absolute path to the parent folder of erllm and adds it to the site-packages directory of the current Python interpreter.  This makes erllm behave as if it were a installed third-party packages.  Yet it also supports code changes without reinstall. |
| [gen_docs.py](erllm/gen_docs.py) |  |
| [utils.py](erllm/utils.py) |  |

## Package: erllm.calibration

| Module | Purpose |
| --- | --- |
| [calibration_plots.py](erllm/calibration/calibration_plots.py) | Performs calibration analysis on language model predictions for different datasets, calculating Brier Score and Expected Calibration Error (ECE).  It generates visualizations of reliability diagrams and saves the calibration metrics in CSV files, organized by model configurations |
| [confidence_hist.py](erllm/calibration/confidence_hist.py) |  |
| [reliability_diagrams.py](erllm/calibration/reliability_diagrams.py) | Third party code from https://github.com/hollance/reliability-diagrams with some small changes. Code is licensed under the MIT license: MIT License  Copyright (c) 2020 M.I. Hollemans  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. |

## Package: erllm.dataset

| Module | Purpose |
| --- | --- |
| [entity.py](erllm/dataset/entity.py) |  |
| [load_ds.py](erllm/dataset/load_ds.py) |  |
| [sample_ds.py](erllm/dataset/sample_ds.py) |  |
| [stats_ds.py](erllm/dataset/stats_ds.py) |  |

## Package: erllm.dataset.dbpedia

| Module | Purpose |
| --- | --- |
| [access_dbpedia.py](erllm/dataset/dbpedia/access_dbpedia.py) | Access the DBPedia SQLite database after it has been created by load_dbpedia.py. |
| [load_dbpedia.py](erllm/dataset/dbpedia/load_dbpedia.py) | Reads raw data from specified paths and loads it into SQLite tables.  The primary tables store DBpedia entities with key-value pairs, and an additional table stores matching pairs. Execute directly to populate the database. |
| [sample_dbpedia.py](erllm/dataset/dbpedia/sample_dbpedia.py) |  |
| [to_ditto.py](erllm/dataset/dbpedia/to_ditto.py) |  |
| [token_blocking.py](erllm/dataset/dbpedia/token_blocking.py) |  |

## Package: erllm.dataset.ditto

| Module | Purpose |
| --- | --- |
| [add_to_ditto_configs.py](erllm/dataset/ditto/add_to_ditto_configs.py) |  |
| [to_ditto.py](erllm/dataset/ditto/to_ditto.py) |  |
| [to_ditto_runner.py](erllm/dataset/ditto/to_ditto_runner.py) |  |

## Package: erllm.discarder

| Module | Purpose |
| --- | --- |
| [discarder.py](erllm/discarder/discarder.py) |  |
| [discarder_eval.py](erllm/discarder/discarder_eval.py) |  |
| [discarder_vis.py](erllm/discarder/discarder_vis.py) |  |

## Package: erllm.discarding_matcher

| Module | Purpose |
| --- | --- |
| [discarding_matcher.py](erllm/discarding_matcher/discarding_matcher.py) |  |
| [discarding_matcher_duration_cmp.py](erllm/discarding_matcher/discarding_matcher_duration_cmp.py) |  |
| [discarding_matcher_runner.py](erllm/discarding_matcher/discarding_matcher_runner.py) | This script runs the discarding matcher algorithm on multiple datasets with different threshold values. It calculates various performance metrics such as accuracy, precision, recall, F1 score, cost, and duration. The results are stored in a pandas DataFrame and saved as a CSV file. |
| [discarding_matcher_tradeoff.py](erllm/discarding_matcher/discarding_matcher_tradeoff.py) | Generate and analyze performance trade-off metrics for the discarding matcher based on F1 decrease thresholds. Reads performance metrics from a CSV file, calculates F1 decrease, relative cost, and relative duration for each dataset and threshold. |
| [discarding_matcher_vis.py](erllm/discarding_matcher/discarding_matcher_vis.py) | This script generates performance comparison plots for the discarding matcher. It reads performance metrics from a CSV file, filters the data based on selected metrics, and creates line plots for each dataset with different configurations, such as all metrics, no cost, and F1 with cost. It also creates plots with showing the performance on all datasets at once. |

## Package: erllm.discarding_selective_matcher

| Module | Purpose |
| --- | --- |
| [discarding_selective_matcher.py](erllm/discarding_selective_matcher/discarding_selective_matcher.py) |  |
| [discarding_selective_matcher_allstats_table.py](erllm/discarding_selective_matcher/discarding_selective_matcher_allstats_table.py) |  |
| [discarding_selective_matcher_contour.py](erllm/discarding_selective_matcher/discarding_selective_matcher_contour.py) |  |
| [discarding_selective_matcher_eval.py](erllm/discarding_selective_matcher/discarding_selective_matcher_eval.py) |  |
| [discarding_selective_matcher_metric_table.py](erllm/discarding_selective_matcher/discarding_selective_matcher_metric_table.py) |  |
| [discarding_selective_matcher_runner.py](erllm/discarding_selective_matcher/discarding_selective_matcher_runner.py) |  |

## Package: erllm.llm_matcher

| Module | Purpose |
| --- | --- |
| [cost.py](erllm/llm_matcher/cost.py) |  |
| [evalrun.py](erllm/llm_matcher/evalrun.py) | Methods for reading run files, deriving classification decisions, and calculating classification and calibration metrics |
| [gpt.py](erllm/llm_matcher/gpt.py) |  |
| [gpt_chat.py](erllm/llm_matcher/gpt_chat.py) |  |
| [llm_matcher.py](erllm/llm_matcher/llm_matcher.py) | Provides functions to evaluate the performance of the LLM Mathcer on a set of run files produced by the OpenAI GPT API.  It calculates various classification metrics, entropies, and calibration results.  The evaluation results are saved as JSON files for individual runs and aggregated into a CSV file for further analysis. |
| [prompt_data.py](erllm/llm_matcher/prompt_data.py) |  |
| [prompts.py](erllm/llm_matcher/prompts.py) |  |

## Package: erllm.selective_classifier

| Module | Purpose |
| --- | --- |
| [selective_classifier.py](erllm/selective_classifier/selective_classifier.py) |  |
| [selective_classifier_runner.py](erllm/selective_classifier/selective_classifier_runner.py) | This script runs the discarding matcher algorithm on multiple datasets with different threshold values. It calculates various performance metrics such as accuracy, precision, recall, F1 score, cost, and duration. The results are stored in a pandas DataFrame and saved as a CSV file. |
| [selective_classifier_tab.py](erllm/selective_classifier/selective_classifier_tab.py) | This script generates performance comparison plots for the discarding matcher. It reads performance metrics from a CSV file, filters the data based on selected metrics, and creates line plots for each dataset with different configurations, such as all metrics, no cost, and F1 with cost. It also creates plots with showing the performance on all datasets at once. |
| [selective_classifier_vis.py](erllm/selective_classifier/selective_classifier_vis.py) | This script generates performance comparison plots for the discarding matcher. It reads performance metrics from a CSV file, filters the data based on selected metrics, and creates line plots for each dataset with different configurations, such as all metrics, no cost, and F1 with cost. It also creates plots with showing the performance on all datasets at once. |

## Package: erllm.selective_matcher

| Module | Purpose |
| --- | --- |
| [random_table.py](erllm/selective_matcher/random_table.py) | This script generates performance comparison plots for the discarding matcher. It reads performance metrics from a CSV file, filters the data based on selected metrics, and creates line plots for each dataset with different configurations, such as all metrics, no cost, and F1 with cost. It also creates plots with showing the performance on all datasets at once. |
| [random_table_with_sd.py](erllm/selective_matcher/random_table_with_sd.py) | This script generates performance comparison plots for the discarding matcher. It reads performance metrics from a CSV file, filters the data based on selected metrics, and creates line plots for each dataset with different configurations, such as all metrics, no cost, and F1 with cost. It also creates plots with showing the performance on all datasets at once. |
| [selective_matcher.py](erllm/selective_matcher/selective_matcher.py) | Defines functions to manually label predictions by selecting the k most uncertain,  k most uncertain negative, and k random predictions from a given set. It applies these labeling strategies to predictions on different datasets,  calculates various classification metrics, and saves the results for comparison. |
| [selective_matcher_runner.py](erllm/selective_matcher/selective_matcher_runner.py) | This script runs the discarding matcher algorithm on multiple datasets with different threshold values. It calculates various performance metrics such as accuracy, precision, recall, F1 score, cost, and duration. The results are stored in a pandas DataFrame and saved as a CSV file. |
| [selective_matcher_vs_base_table.py](erllm/selective_matcher/selective_matcher_vs_base_table.py) | This script generates performance comparison plots for the discarding matcher. It reads performance metrics from a CSV file, filters the data based on selected metrics, and creates line plots for each dataset with different configurations, such as all metrics, no cost, and F1 with cost. It also creates plots with showing the performance on all datasets at once. |
