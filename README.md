# GitHub repository structure
```
├── Data/                       # Train and test data
│   ├── net-load-forecasting-during-soberty-period
|        └── test.csv
|        └── train.csv
|   └── sample_submission.csv     
│
├── analysis/                   # Model analysis and hyperparameter selection
│   └── cv_rbf.py
│   └── rbf_viz.py
│   └── tests.py
│   └── val_scores_linear.py
├── img/                        # Related graphics and images
├── requirements.txt
├── data_utils.py               # Contains feature engineering and Kalman filter
├── kalman_interactive.py       # Notebook to observe the positive effects of Kalman filter
├── model_interactive.py        # Model code + plots to interpret the model
├── model_submission.py         # Model code that gives submission_final.csv
├── submission_final.csv        # Ready final submission
└── README.md
```

# Analysis folder description

* ```cv_rbf.py``` contains ```TimeSeriesSplit```-driven cross-validation for 5 hyperparameters: L1-regularization, number of Fourier features, 3 RBF hyperparameters

* ```rbf_viz``` contains visualization of approximation of sine function by RBF function (used only for presentation)

* ```tests.py``` contains correlation calculations over certain features and some data analysis

* ```val_scores_linear.py``` contains benchmark of three linear models: default, default + some features, model with RBF features

# Note on reproducibility

In notebooks, we added requirement to have scikit-learn version=1.8.0 since our machine uses this version
All the random seeds are fixed where used