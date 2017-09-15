## Installation Instructions

```bash
$ pip3 install --upgrade pip virtualenv virtualenvwrapper

$ pip3 install --upgrade jupyter matplotlib numpy panadas scipy scikit-learn statsmodels

$ jupyter notebook
```

#### Configure matplotlib

```bash
# Required for mac os x - https://stackoverflow.com/a/21789908/1216965
$ echo "backend : TkAgg" > ~/.matplotlib/matplotlibrc
# or
$ echo "backend : Agg" > ~/.matplotlib/matplotlibrc

```

#### Configure virtualenv and run notebook

```bash
$ mkvirtualenv --system-site-packages -p python3 machine_learning

$ workon machine_learning
$ cd learning_stats
$ jupyter notebook
```