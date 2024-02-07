
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
