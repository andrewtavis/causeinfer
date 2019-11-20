![](https://github.com/andrewtavis/causeinfer/blob/master/resources/causeinfer_logo.png)
Causal inference/uplift in Python.

[![GitHub](https://img.shields.io/github/license/andrewtavis/causeinfer.svg)](https://github.com/andrewtavis/causeinfer/LICENSE)

[Application](#application) •
[Included Datasets](#included-datasets) •
[Contribute](#contribute) •
[References](#references) •
[License](https://github.com/andrewtavis/causeinfer/LICENSE)

## Getting Started
Latest release version: 0.0.1

### Installation
```bash
pip install causeinfer
```

## Application

### Causal inference algorithms:
#### 1. The Two Model Approach
- Separate models for treatment and control groups are trained and combined to derive average treatment effects.

#### 2. Interaction Term Approach - Lo 2002
- An interaction term between treatment and covariates is added to the data to allow for a basic single model application.

#### 3. Response Transformation Approach - Lai 2006; Kane, Lo and Zheng 2014
- Units are categorized to allow for the derivation of treatment effected covariates through classification.\

#### 4. Generalized Random Forest - Athey, Tibshirani, and Wager 2019
- An application of an honest causalaity based splitting random forest.

### Evaluation metrics:
#### 1. Qini and AUUC Scores
- Comparisons across stratefied, ordered treatment response groups are used to derive model efficiency

#### 2. GRF Confidence Intervals
- Confidence intervals are created using GRF's standard deviation across trials

## Included Datasets
- [Hillstrom Email Marketing](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
- [Mayo Clinic PBC](https://www.mayo.edu/research/documents/pbchtml/DOC-10027635)
- [IFMR Microfinance](https://www.aeaweb.org/articles?id=10.1257/app.20130533)

## Contribute
#### Contributions are more than welcome!
- [Examples:](https://github.com/andrewtavis/causeinfer/examples) share more applications
- [Issues:](https://github.com/andrewtavis/causeinfer/issues? add, or see what's to be done

## Similar Packages
### The following are similar packages/modules to causeinfer:
#### Python:
- https://github.com/uber/causalml
- https://github.com/Minyus/causallift
- https://github.com/jszymon/uplift_sklearn
- https://github.com/duketemon/pyuplift
- https://github.com/microsoft/EconML
- https://github.com/wayfair/pylift/

#### Other Languages:
- https://github.com/grf-labs/grf (R/C++)
- https://github.com/imbs-hl/ranger (R/C++)
- https://github.com/soerenkuenzel/causalToolbox/blob/a06d81d74f4d575a8b34dc6b718db2778cfa0be9/R/XRF.R (R/C++)
- https://github.com/soerenkuenzel/forestry (R/C++)
- https://github.com/cran/uplift/tree/master/R (R)

## References
<details><summary>Full list of theoretical references</summary>
<p>
- 

</p>
</details>