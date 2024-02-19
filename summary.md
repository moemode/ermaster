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

- **embed-k:** The random selection results in different attributes for entities in a pair. At the same time, it preserves all the information in the attribute values that may be crucial for the matching decision.

- **misfield-k:** Generates misfielded values by assigning attribute values to a wrong attribute name.

The inclusion of data errors leads to a deterioration of all metrics in all cases except precision for embed-1. 

The simulated data errors have a greater negative impact on recall than precision. 

The classification performance, as measured by F1, deteriorates for entity pairs with differing attributes and misfielded values (embed-1, embed-50%).

In conclusion, misfielded values and a mixed schema with embedded values are a considerable challenge for the *LLM* (*gpt-3.5-turbo-instruct*).


## 5.3 Discarding Matcher

### 5.3.1 Evaluation of Discarder
Desc:
The discarding matcher comprises a discarder placed in front of a Large Language Model (LLM) matcher. The discarder calculates the similarity between two profiles using Jaccard and overlap set-based similarity measures, and cosine similarity from static GloVe word-embeddings. Only profile pairs with similarity above a certain threshold are forwarded to the LLM for classification.

Datasets exhibit two patterns in the RR-PC trade-off: an "upside-down hockey stick" pattern signaling high potential for recall with substantial pair reduction, and a concave curve indicating a steeper PC decrease with RR increase, which is less desirable.

Findings:
Overall, set-based similarities outperform cosine similarity on embedding vectors. The overlap-based discarder shows the best overall performance.

It is possible to discard a substantial amount (>50\%) of pairs while maintaining a PC and thus potential recall of the subsequent matching step 
above 80\%. For the majority of datasets this trade-off is even much better.

### 5.3.2 Evaluation of Discarding Matcher

#### Time required by Discarder
This shows that the run time of the discarder is negligible compared to the run time of the \gls{llm}.

The discarder enables a large net time saving by being fast itself and avoiding the LLM computation on the discarded pairs.

#### Trade-off between Cost or Time and Classification Performance

the discarding matcher works best on the semi-structured DBPedia dataset and provides large cost and time savings on all datasets.

In total, six datasets have a cost reduction of over 80\% when allowing for a F1 reduction of 2.5\%.

Even on datasets where discarding works worse / less favorable RR-PC trade-offs (Abt-Buy and Walmart-Amazon), 
considerable  cost reductions are possible.
For these two datasets, cost reductions of about 78\% and 46\% are possible  when allowing for an F1 reduction of 10\%.
The discarding matcher is an efficient way to integrate an \gls{llm} for entity matching

### 5.4 Selective Matcher


For the confidence estimator we use the normalized maximum softmax probability

#### 5.4.1 Confidence Estimator Calibration

The Expected Calibration Error (ECE) is employed to measure the calibration quality, with lower ECE values indicating better calibration. Across various datasets, the ECE values ranged from 2.38\% to 4.35\%, suggesting a generally good calibration.

#### 5.4.2 Estimating Performance

We create a soft confusion matrix by leveraging the confidence values for estimating different performance metrics (accuracy, precision, recall, and F1 score).
the accuracy estimates were close to actual values, with a maximum error of 4.27\%. 
However, the precision, recall, and F1 estimates had large relative errors making them useless for performance estimation

#### 5.4.3 Selective Classification

Whereas the LLM makes a binary decision for all profile pairs in the LLM
matcher, the selective LLM matcher abstains from a decision when it is unconfident.
The abstained on profile pairs remain unlabeled, and we evaluate performance based
on labeled pairs only.


Datasets showed mixed results when applying selective classification. 
On some datasets, reducing coverage (labeling fewer predictions) improves the F1 score on some datasets but has the exact opposite effect on others.

Selective classification can be used for extracting near-certain matches which could serve as ground-truth to train other models, for manual inspection, or enhancing the LLM's responses.

#### 5.4.4 Selective Matcher
In it, the LLM abstains from
predictions below a certain confidence threshold and a human labels the abstained on pairs correctly in the final step

the selective matcher enables large improvements in classification performance as measured by F1 over the basic LLM Matcher across datasets.

Manual labeling of these low-confidence pairs is compared against randomly labeling an equivalent number of pairs.

the selective matcher is much more effective in improving classification performance than random labeling.

Indeed, low confidences are a good indicator for where to focus the manual labeling effort and the selective matcher is an effective way to integrate a \gls{llm} for high quality entity matching