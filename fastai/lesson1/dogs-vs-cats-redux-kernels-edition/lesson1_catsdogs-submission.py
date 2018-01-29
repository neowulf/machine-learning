
# coding: utf-8

# ## Preamble

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


# ## Project Setup

# In[2]:


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


# ## Model preparation

# ### Finetune the keras model
# * Pop the last layer, freeze all layers, add a softmax layer and update set of classes

# In[3]:


vgg16 = KerasVgg16(WEIGHTS_DIR)
vgg16.create_model(learning_rate = 0.01, ttl_outputs=2)
vgg16.model.summary()


# ### Fit the keras model
# 1. Train the updated keras model

# In[4]:


train_generator = vgg16.generator(TRAIN_DIR, 250)
valid_generator = vgg16.generator(CROSSVALID_DIR, 250)

# 581 seconds per epoch when use_multiprocessing=False
# vgg16.finetune(train_generator, valid_generator, epochs=3)

# use_multiprocessing=True - couldn't pickle..doesn't work with tensorflow?
# vgg16.finetune(train_generator, valid_generator, epochs=2, use_multiprocessing=True)

# bcolz approach
vgg16.save_generator(train_generator, 'sample_train')
vgg16.save_generator(valid_generator, 'sample_valid')


# In[ ]:


trn = vgg16.load_bcolz_generator('sample_train', self.onehot(train_generator), 35)
valid = vgg16.load_bcolz_generator('sample_valid', self.onehot(valid_generator), 35)


# In[ ]:


vgg16.finetune(trn, valid, epochs=2)


# ### Save and load the model after couple of epochs

# In[ ]:


# filename='3_epochs_finetune_96'
# vgg16.save_weights(filename)
# vgg16.load_weights(filename)


# ## Perform predictions

# ## Kaggle Submit

# ### Prepare csv file and Submit
# 
