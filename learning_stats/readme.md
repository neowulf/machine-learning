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

#### Configure virtualenv

```bash
$ mkvirtualenv --system-site-packages -p python3 machine_learning

$ workon machine_learning

# doesn't influence jupyter notebook
$ add2virtualenv ThinkStats2/code/
```

#### Run notebook

```bash
$ workon machine_learning
$ cd learning_stats
$ jupyter notebook
```

#### Prepend to notebook

```python 
# Configures matplotlib
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# Allows for interactive shell - outputs all non variable statements
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# Allows for importing ThinkStats2 Code
import sys
import os 
sys.path.insert(0, os.getcwd() + '/ThinkStats2/code')

# Load Files
datafile_base= "ThinkStats2/code/2002FemPreg"
preg = nsfg.ReadFemPreg(dct_file =datafile_base + ".dct", dat_file =datafile_base + ".dat.gz")

datafile_base= "ThinkStats2/code/2002FemResp"
resp = nsfg.ReadFemResp(dct_file =datafile_base + ".dct", dat_file =datafile_base + ".dat.gz")
```
