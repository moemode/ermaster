# Summary

## 5.2 Entity Matching using LLM Matcher

### 5.2.1 Performance

Overall, while *gpt-3.5-turbo-instruct* achieves high precision in entity matching tasks, it often does so at the cost of low recall and low F1 scores across multiple datasets. The semi-structured DBpedia set ranks fourth in terms of F1 with perfect precision and mediocre recall. Given that it is placed higher than many other datasets, its semi-structured nature does not seem to pose extraordinary problems for the *LLM*. However, given the observed variability in classification performance and the fact that the F1 score is quite low for five out of nine datasets, we find that *gpt-3.5-turbo-instruct* is not a turnkey solution for entity matching across diverse datasets.

### 5.2.2 A case of prompt sensitivity

However, we observe a drastically different trade-off between precision and recall between two prompt designs with only a difference in the prescribed answer format (base vs. Hash). This shows that *gpt-3.5-turbo-instruct* is brittle with respect to prompt design in this classification use-case.

### 5.2.3 Cost

The total cost for all datasets is $2.22. There is a need to integrate *LLM* in a cost-efficient way.

### 5.2.4 Entity Serialization Ablations

#### Attribute Names vs Only Values

The *LLM* is unduly sensitive to the attribute order, as a randomized attribute order leads to lower classification performance than a fixed order. Including attribute names improves classification performance over serializing only attribute values in almost all cases.

#### Mixed Attributes and Data Errors

- **Embed-k:** The random selection results in different attributes for entities in a pair. At the same time, it preserves all the information in the attribute values that may be crucial for the matching decision.

- *misfield-k:* Generates misfielded values by assigning attribute values to a wrong attribute name. The inclusion of data errors leads to a deterioration of all metrics in all cases except precision for embed-1. The simulated data errors have a greater negative impact on recall than precision. The classification performance, as measured by F1, deteriorates for entity pairs with differing attributes and misfielded values (embed-1, embed-50%).

In conclusion, misfielded values and a mixed schema with embedded values are a considerable challenge for the *LLM* (*gpt-3.5-turbo-instruct*).
