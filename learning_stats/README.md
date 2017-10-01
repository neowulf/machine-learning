## Installation Instructions

```bash

conda create -n machine_learning python=3.5 jupyter matplotlib numpy pandas scipy scikit-learn statsmodels

conda activate machine_learning
conda env export > conda-environment.yml
```

#### Configure matplotlib

```bash
# Required for mac os x - https://stackoverflow.com/a/21789908/1216965
$ echo "backend : TkAgg" > ~/.matplotlib/matplotlibrc
# or
$ echo "backend : Agg" > ~/.matplotlib/matplotlibrc

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
