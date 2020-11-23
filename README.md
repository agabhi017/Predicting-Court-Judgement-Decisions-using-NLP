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
  Article#  | Violations | Not Violations
----------- | ---------- | --------------
Article 3   | 591        | 560
Article 5   | 509        | 437
Article 6   | 754        | 565
Article 8   | 411        | 351

## Results
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

## Full Report
The full report is available [here](https://github.com/agabhi017/Predicting-Court-Judgement-Decisions-using-NLP/blob/master/NLP_Report.pdf)

## Presentation
The presentation is available [here](https://agabhi017.github.io/Predicting-Court-Judgement-Decisions-using-Natural-Language-Processing/)
