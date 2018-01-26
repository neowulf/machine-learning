
# coding: utf-8

# In[1]:


# Allows for interactive shell - outputs all non variable statements
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# https://ipython.org/ipython-doc/3/config/extensions/autoreload.html
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
get_ipython().run_line_magic('aimport', 'keras_vgg16')
from keras_vgg16 import *


# In[5]:


import os
import shutil
from glob import glob
np.random.seed(10)

current_dir = os.getcwd()
DATASET_DIR=os.path.join(current_dir, 'dataset')
CROSSVALID_DIR=os.path.join(DATASET_DIR, 'cross_valid')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
CROSSVALID_DIR = os.path.join(DATASET_DIR, 'cross_valid')
SAMPLE_DIR = os.path.join(DATASET_DIR, 'sample')

SAMPLE_TRAIN_DIR=os.path.join(SAMPLE_DIR, 'train')
SAMPLE_CROSSVALID_DIR=os.path.join(SAMPLE_DIR, 'cross_valid')

WEIGHTS_DIR = os.path.join(current_dir, 'weights')


# In[3]:


vgg16 = KerasVgg16(WEIGHTS_DIR)
vgg16.create_model(learning_rate = 0.01, ttl_outputs=2)
vgg16.model.summary()


# In[ ]:


train_generator = vgg16.generator(TRAIN_DIR, 35)
valid_generator = vgg16.generator(CROSSVALID_DIR, 35)

vgg16.finetune('sample', train_generator, valid_generator, epochs=3)

