# Predicting Court Judgement Decisions using Natural Language Processing

## Introduction
Legal institutions in most countries suffer from significant delay due to large
number of cases. This is not only an issue in law & ethics but a research problem which comes under the purview of legal engineering. With modern tools
and computational abilities we can look into using Natural Language Processing(NLP) as a guiding mechanism for legal systems and influence the productivity of our over-burdened courts.

## Case for India
The courts of india are relatively overburdened. The New Delhi’s High Court
(2009) observed that the existing backlog of cases would take another 466 years
to come to a verdict! The Hindu estimates the number of cases pending with
Indian courts to be around 30 million. The Law Commission, 1987, estimated
that there are 10 judges for every million population of India. With increasing
population the cases with the courts has increased drastically while the judge to
population ratio did not increase to the recommended level. All this together
stanches the flow of justice.


## Objective
Predicting the outcomes of cases which are under the jurisdiction of European
Court of Human Rights, regarding violations of Articles 3,5,6,8. Output will
be a binary vector of Violation and No violation.

## Data
The number of transacripts available for each article and violation/non-violation are summarised below.
  Article#  | Violations | Not Violations
----------- | ---------- | --------------
Article 3   | 591        | 560
Article 5   | 509        | 437
Article 6   | 754        | 565
Article 8   | 411        | 351

## Classification Models
The following models were used for the binary classification :
* Neural Network
* Support Vector Machines
* Prediction Scores\
Using the weights calculated, we formulate a predictive score for each paragraph
(via aggregation of the weights of constituent words). As with the word weights,
a positive paragraph score suggests article violation while a negative score suggests no violation.
Calculating the prediction scores for all paragraphs in a document we use three
approaches to interpret the aggregate score to a single prediction :
  * Min-Max : We sort the prediction scores for each paragraphs and use the sum of min and max
  * Aggregate sum : We take a linear summation of prediction scores of all paragraphs. A positive sum predicts a violation and vice-versa
  * Weighted sum : We take a weighted mean of the prediction paragraph score using word size as the weight of each paragraph

## Results
The following table summarizes the accuracies with respect to the topmost(most frequent) features corresponding to each of the three articles(3, 6 & 8) used in [1]. 
Moreover it can be seen that the accuracies in general are low for all the articles
and subsections, this can be attributed to the fact that the topmost features are
not so predictive of the violation/non-violation of the articles. Adding to this
the data set used in [1] was very less and hence the weights obtained might be
questionable too.
Case Structure |Metric| Article 3 |Article 6| Article 8
---------------|------|-----------|---------|----------
Procedure|Max/min| 65.7| 58.32| 53.05
||Sum Compare |66.78 |57.18| 50.13
||Size Weighted Sum| 66.51| 55.74| 49.73
Facts|Max/min| 60.64 |56.92| 54.88
||Sum Compare| 60.99 |56.54| 54.19
||Size Weighted Sum| 60.18| 56.39| 53.92
The Law|Max/min| 54.59 |56.62| 51.85
||Sum Compare |53.87 |57.22 |48.83
||Size Weighted Sum| 53.96| 56.92| 48.84
Full Doc|Max/min| 57.16| 56.86| 53.14
||Sum Compare |54.12| 57.24| 50.26
||Size Weighted Sum| 55.25| 57.31| 49.88

The accuracies of subsections and different metrics using the most predictive features for violation and non-violation of articles and their
corresponding weights obtained by training a model using SVM classifier and
a linear kernel are summarised in the table below :
Case Structure | Metric | Article 3 | Article 5 | Article 6 | Article 8
|---------------|--------|-----------|-----------|-----------|----------
|Procedure |Max/min | 81.22 | 80.86 | 81.67 |71.07
||Sum Compare | 81.57 | 81.08 | 81.90 | 71.27
||Size Weighted Sum | 81.40| 80.65| 81.52| 72.07
Facts |Max/min |66.87| 71.80| 68.34| 71.11
||Sum Compare| 68.59| 74.33| 71.23| 75.37
||Size Weighted Sum |68.86 |74.10 |71.68| 73.31
The Law| Max/min| 73.12| 71.47 |72.07| 70.79
||Sum Compare |78.35| 76.74| 78.38| 76.44
||Size Weighted Sum| 75.02 |73.76 |77.54| 74.49
Full Doc| Max/min| 72.98| 75.89| 79.22| 70.99
||Sum Compare| 75.15| 73.28| 80.00 |74.14
||Size Weighted Sum| 72.54| 79.06 |79.57 |74.27

These results indicate that the SVM classifier was able to identify more predictive features/topics as compared to manual identification.


The table below summarizes the performance of the ”tf-idf” representation using Feed forward neural networks and SVM as classifiers.
Classifier| Metric |Article 3 |Article 5 |Article 6 |Article 8
----------|--------|----------|----------|----------|---------
NN| Procedure |92.48 |92.92 |93.41 |92.97
||Facts| 90.92 |90.05| 91.69 |91.00
||The Law |91.80 |91.23 |92.77 |92.04
||Full Doc |92.16 |92.24 |93.60 |93.12
SVM| Procedure |96.31 |96.41 |95.74 |94.28
||Facts |92.96 |92.63 |94.14 |91.76
||The Law |94.86 |94.04| 95.05 |93.41
||Full Doc |94.17 |94.82 |95.67 |94.48

## Discussion/Conclusion
In contrast to the analysis done by Aletras 2016 which concluded that relevant
facts has the highest predictive performance that resonates with he principles
of legal realism, we find that with more enriched word representations like word
specific prediction weights and tfidf features, Procedure outperforms other
sections. Various explanation can fit this observation. The most relevant one
seems to be that the section Procedure has the most concise description of the facts of the case, hence the most weighted words happen to be in this section,
that is the section is fact dense. other explanations could be that outcome for
a case is biased by the pre judicial treatment of the lodged complaints and the
ruling of the domestic courts are good predictors of the outcome.

## References
1. Nikolaos Aletras, Dimitrios Tsarapatsanis, Daniel Preot¸iuc-Pietro, and
Vasileios Lampos. Predicting judicial decisions of the european court of
human rights: A natural language processing perspective. PeerJ Computer
Science, 2:e93, 2016.
2. Reed C Lawlor. What computers can do: Analysis and prediction of judicial
decisions. American Bar Association Journal, pages 337–344, 1963.
3. Benjamin E Lauderdale and Tom S Clark. The supreme court’s many median
justices. American Political Science Review, 106(4):847–866, 2012.

## Full Report
The full report is available [here](https://github.com/agabhi017/Predicting-Court-Judgement-Decisions-using-NLP/blob/master/NLP_Report.pdf)

## Presentation
The presentation is available [here](https://agabhi017.github.io/Predicting-Court-Judgement-Decisions-using-Natural-Language-Processing/)
