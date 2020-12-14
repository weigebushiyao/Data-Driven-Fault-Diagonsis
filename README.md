Documentation
=============
**General Description:**

This project is a data-driven fault diagnosis for industrial process (icing condition of wind turbines) programmed with Python. 

Project mainly includes three parts:
1. preprocessing part
2. feature extraction part
3. data modeling part

Dependencies
-------------
**Language:**  python

package             | version       
------------------- | --------------
*numpy*|1.18.2
*pandas*|0.0.97
*tensorflow*|2.0.0
*matplotlib*|3.3.3
*sklearn*|0.0
*seaborn*|0.11.0
*itertools*|-
*imblearn*|-

Project Architecture
-------------
```buildoutcfg
    CNN.py                      // Convolutional Neural Network classifier
    data_integration.py         // data integration module
    Document.pdf                // technical documentation of this project
    fisher_discrimination.py    // fisher discrimination classifier
    PCA.py                      // PCA feature extraction module
    RandomForest.py             // random forest classifier
    README.md                   // help
    SMOTE_sampling.py           // downsampling module
```