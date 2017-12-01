1. Update ~/.bashrc
```bash
source activate root
cd /home/ubuntu/machine-learning/fastai
```
1. Update packages
```bash
pip install --upgrade pip
pip install kaggle-cli
sudo apt install unzip tree htop
```

1. Update the jupyter configuration at `~/.jupyter/jupyter_notebook_config.py`
```ini
c.NotebookApp.ip = '127.0.0.1'
c.NotebookApp.port = 8889
c.NotebookApp.open_browser = False

c.NotebookApp.password = u'sha1:7c4c2f4e6058:71f644cc062dcf2533c6f205993ebe04d6e8406c'
```

1. Get the [kaggle dataset](http://wiki.fast.ai/index.php/Kaggle_CLI)
