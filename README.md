<div align="center">
  <a href="https://github.com/andrewtavis/causeinfer"><img src="https://raw.githubusercontent.com/andrewtavis/causeinfer/master/resources/causeinfer_logo.png"></a>
</div>

- ### Machine learning based causal inference/uplift in Python
------------------------------------------------------

[![PyPI Version](https://badge.fury.io/py/causeinfer.svg)](https://pypi.org/project/causeinfer/)
[![Python Version](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue.svg)](https://pypi.org/project/causeinfer/)
[![GitHub](https://img.shields.io/github/license/andrewtavis/causeinfer.svg)](https://github.com/andrewtavis/causeinfer/blob/master/LICENSE)

[Application](#application) •
[Included Data and Examples](#included-data-and-examples) •
[Contribute](#contribute) •
[References](#references)

# Installation
```bash
pip install causeinfer
```

# Application

### Causal inference algorithms:
<details><summary><strong>1. The Two Model Approach<strong></summary>
<p>

- Separate models for treatment and control groups are trained and combined to derive average treatment effects.

</p>
</details>

<details><summary><strong>2. Interaction Term Approach - Lo 2002<strong></summary>
<p>

- An interaction term between treatment and covariates is added to the data to allow for a basic single model application.

</p>
</details>

<details><summary><strong>3. Class Transformation Approaches - Lai 2006; Kane, Lo and Zheng 2014<strong></summary>
<p>

<div align="center">
  <img src="https://raw.githubusercontent.com/andrewtavis/causeinfer/master/resources/new_known_unknown_classes.png" width="720" height="405">
</div>

- Units are categorized into two or four classes to derive treatment effects from favorable class attributes.

</p>
</details>

<details><summary><strong>4. Generalized Random Forest - Athey, Tibshirani, and Wager 2019 (in progress)<strong></summary>
<p>

- A wrapper application of honest causalaity based splitting random forests - via the R/C++ [grf](https://github.com/grf-labs/grf).

</p>
</details>

### Evaluation metrics:
#### 1. Qini and AUUC Scores
Comparisons across stratefied, ordered treatment response groups are used to derive model efficiency.

#### 2. GRF Confidence Intervals
Confidence intervals are created using GRF's honesty based, Gaussian assymptotic forest summations.

# Included Data and Examples
<details><summary><strong>Business Analytics<strong></summary>
<p>

- [Hillstrom Email Marketing](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
  - Is directly downloaded and formatted with CauseInfer [(see script)](https://github.com/andrewtavis/causeinfer/blob/master/causeinfer/data/hillstrom.py).
  - [Example notebook](https://github.com/andrewtavis/causeinfer/blob/master/examples/marketing_hilstrom.ipynb) (in progress).

- [Criterio Uplift](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
  - Download and formatting script in progress.
  - Example notebook to follow.

</p>
</details>

<details><summary><strong>Medical Trials<strong></summary>
<p>

- [Mayo Clinic PBC](https://www.mayo.edu/research/documents/pbchtml/DOC-10027635)
  - Is directly downloaded and formatted with CauseInfer [(see script)](https://github.com/andrewtavis/causeinfer/blob/master/causeinfer/data/mayo_pbc.py).
  - [Example notebook](https://github.com/andrewtavis/causeinfer/blob/master/examples/medical_mayo_clinic_pbc.ipynb) (in progress).

- [Pintilie Tamoxifen](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470870709)
  - Accompanied the linked text, but is now unavailable. It is provided in the [datasets directory](https://github.com/andrewtavis/causeinfer/tree/master/causeinfer/data/datasets) for direct download.
  - Formatting script in progress.
  - Example notebook to follow.

</p>
</details>

<details><summary><strong>Socio-economic Analysis<strong></summary>
<p>

- [CMF Microfinance](https://www.aeaweb.org/articles?id=10.1257/app.20130533)
  - Accompanied the linked text, but is now unavailable. It is provided in the [datasets directory](https://github.com/andrewtavis/causeinfer/tree/master/causeinfer/data/datasets) for direct download.
  - Is formatted with CauseInfer [(see script)](https://github.com/andrewtavis/causeinfer/blob/master/causeinfer/data/cmf_microfinance.py).
  - [Example notebook](https://github.com/andrewtavis/causeinfer/blob/master/examples/socio_econ_center_for_microfinance.ipynb) (in progress).

- [Lalonde Job Training](https://users.nber.org/~rdehejia/data/.nswdata2.html)
  - Download and formatting script in progress.
  - Example notebook to follow.

</p>
</details>

<details><summary><strong>Simmulated Data<strong></summary>
<p>

- Work is currently being done to add a data generator, thus allowing for theoretical tests with known treatmet effects. 
- Example notebook to follow.

</p>
</details>

# Contribute
- [Examples](https://github.com/andrewtavis/causeinfer/tree/master/examples): share more applications
- [Issues](https://github.com/andrewtavis/causeinfer/issues?): suggestions and improvements more than welcome!

# Similar Packages
<details><summary><strong>Similar packages/modules to causeinfer<strong></summary>
<p>

<strong>Python</strong>

- https://github.com/uber/causalml
- https://github.com/Minyus/causallift
- https://github.com/jszymon/uplift_sklearn
- https://github.com/duketemon/pyuplift
- https://github.com/microsoft/EconML
- https://github.com/Microsoft/dowhy
- https://github.com/wayfair/pylift/

<strong>Other Languages</strong>

- https://github.com/grf-labs/grf (R/C++)
- [https://github.com/soerenkuenzel/causalToolbox/X-Learner](https://github.com/soerenkuenzel/causalToolbox/blob/a06d81d74f4d575a8b34dc6b718db2778cfa0be9/R/XRF.R) (R/C++)
- https://github.com/xnie/rlearner (R)

</p>
</details>

# References
<details><summary><strong>Full list of theoretical references<strong></summary>
<p>

<strong>Big Data and Machine Learning</strong> 

- Athey, S. (2017). Beyond prediction: Using big data for policy problems. Science, Vol. 355, No. 6324, February 3, 2017, pp. 483-485.
- Athey, S. & Imbens, G. (2015). Machine Learning Methods for Estimating Heterogeneous Causal Effects. Draft version submitted April 5th, 2015, arXiv:1504.01132v1, pp. 1-25.
- Athey, S. & Imbens, G. (2019). Machine Learning Methods That Economists Should Know About. Annual Review of Economics, Vol. 11, August 2019, pp. 685-725.
- Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, Vol. 21, No. 1, February 1, 2018, pp. C1–C68.
- Mullainathan, S. & Spiess, J. (2017). Machine Learning: An Applied Econometric Approach. Journal of Economic Perspectives, Vol. 31, No. 2, Spring 2017, pp. 87-106.

<strong>Causal Inference</strong> 

- Athey, S. & Imbens, G. (2017). The State of Applied Econometrics: Causality and Policy Evaluation. Journal of Economic Perspectives, Vol. 31, No. 2, Spring 2017, pp. 3-32.
- Athey, S.,  Tibshirani, J. & Wager, S. (2019) Generalized random forests. The Annals of Statistics, Vol. 47, No. 2 (2019), pp. 1148-1178.
- Athey, S. & Wager, S. (2019). Efficient Policy Learning. Draft version submitted on 9 Feb 2017, last revised 16 Sep 2019, arXiv:1702.02896v5, pp. 1-10.
- Banerjee, A, et al. (2015) The Miracle of Microfinance? Evidence from a Randomized Evaluation. American Economic Journal: Applied Economics, Vol. 7, No. 1, January 1, 2015, pp. 22-53.
- Ding, P. & Li, F. (2018). Causal Inference: A Missing Data Perspective. Statistical Science, Vol. 33, No. 2, 2018, pp. 214-237.
- Farrell, M., Liang, T. & Misra S. (2018). Deep Neural Networks for Estimation and Inference: Application to Causal Effects and Other Semiparametric Estimands. Draft version submitted December 2018, arXiv:1809.09953, pp. 1-54.
- Gutierrez, P. & Gérardy, JY. (2016). Causal Inference and Uplift Modeling: A review of the literature. JMLR: Workshop and Conference Proceedings 67, 2016, pp. 1–14.
- Hitsch, G J. & Misra, S. (2018). Heterogeneous Treatment Effects and Optimal Targeting Policy Evaluation. January 28, 2018, Available at SSRN: ssrn.com/abstract=3111957 or dx.doi.org/10.2139/ssrn.3111957, pp. 1-64. 
- Powers, S. et al. (2018). Some methods for heterogeneous treatment effect estimation in high dimensions. Statistics in Medicine, Vol. 37, No. 11, May 20, 2018, pp. 1767-1787.
- Rosenbaum, P. & Rubin, D. (1983). The central role of the propensity score in observational studies for causal effects. Biometrika, Vol. 70, pp. 41-55.
- Sekhon, J. (2007). The Neyman-Rubin Model of Causal Inference and Estimation via Matching Methods. The Oxford Handbook of Political Methodology, Winter 2017, pp. 1-46.
- Wager, S. & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. Journal of the American Statistical Association, Vol. 113, 2018 - Issue 523, pp. 1228-1242.

<strong>Uplift</strong> 

- Devriendt, F. et al. (2018). A Literature Survey and Experimental Evaluation of the State-of-the-Art in Uplift Modeling: A Stepping Stone Toward the Development of Prescriptive Analytics. Big Data, Vol. 6, No. 1, March 1, 2018, pp. 1-29. Codes found at: data-lab.be/downloads.php.
- Hansotia, B. & Rukstales, B. (2002). Incremental value modeling. Journal of Interactive Marketing, Vol. 16, No. 3, Summer 2002, pp. 35-46.
- Haupt, J., Jacob, D., Gubela, R. & Lessmann, S. (2019). Affordable Uplift: Supervised Randomization in Controlled Experiments. Draft version submitted on October 1, 2019, arXiv:1910.00393v1, pp. 1-15.
- Jaroszewicz, S. & Rzepakowski, P. (2014). Uplift modeling with survival data. Workshop on Health Informatics (HI-KDD) New York City, August 2014, pp. 1-8. 
- Jaśkowski, M. & Jaroszewicz, S. (2012). Uplift modeling for clinical trial data. In: ICML, 2012, Workshop on machine learning for clinical data analysis. Edinburgh, Scotland, June 2012, 1-8.
- Kane, K.,  Lo, VSY. & Zheng, J. (2014). Mining for the truly responsive customers and prospects using true-lift modeling: Comparison of new and existing methods. Journal of Marketing Analytics, Vol. 2, No. 4, December 2014, pp 218–238.
- Lai, L.Y.-T. (2006). Influential marketing: A new direct marketing strategy addressing the existence of voluntary buyers. Master of Science thesis, Simon Fraser University School of Computing Science, Burnaby, BC, Canada, pp. 1-68.
- Lo, VSY. (2002). The true lift model: a novel data mining approach to response modeling in database marketing. SIGKDD Explor 4(2), pp. 78–86.
- Lo, VSY. & Pachamanova, D. (2016). From predictive uplift modeling to prescriptive uplift analytics: A practical approach to treatment optimization while accounting for estimation risk. Journal of Marketing Analytics Vol. 3, No. 2, pp. 79–95.
- Radcliffe N.J. & Surry, P.D. (1999). Differential response analysis: Modeling true response by isolating the effect of a single action. In Proceedings of Credit Scoring and Credit Control VI. Credit Research Centre, University of Edinburgh Management School.
- Radcliffe N.J. & Surry, P.D. (2011). Real-World Uplift Modelling with Significance-Based Uplift Trees. Technical Report TR-2011-1, Stochastic Solutions, 2011, pp. 1-33. 
- Rzepakowski, P. & Jaroszewicz, S. (2012). Decision trees for uplift modeling with single and multiple treatments. Knowledge and Information Systems, Vol. 32, pp. 303–327.
- Rzepakowski, P. & Jaroszewicz, S. (2012). Uplift modeling in direct marketing. Journal of Telecommunications and Information Technology, Vol. 2, 2012, pp. 43–50.
- Rudaś, K. & Jaroszewicz, S. (2018). Linear regression for uplift modeling. Data Mining and Knowledge Discovery, Vol. 32, No. 5, September 2018, pp. 1275–1305.
- Sołtys, M., Jaroszewicz, S. & Rzepakowski, P. (2015). Ensemble methods for uplift modeling. Data Mining and Knowledge Discovery, Vol. 29, No. 6, November 2015,  pp. 1531–1559.

</p>
</details>