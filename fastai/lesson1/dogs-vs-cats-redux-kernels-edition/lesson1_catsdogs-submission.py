
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


# ## Model preparation

# ### Finetune the keras model
# * Pop the last layer, freeze all layers, add a softmax layer and update set of classes

# In[2]:


vgg16 = KerasVgg16(WEIGHTS_DIR)
vgg16.create_model(learning_rate = 0.01, ttl_outputs=2)
vgg16.model.summary()


# In[3]:


train_dir = SAMPLE_TRAIN_DIR
crossvalid_dir = SAMPLE_CROSSVALID_DIR

train_generator = vgg16.generator(train_dir, 1, shuffle=False)
vgg16.save_bcolz_generator(train_generator, train_dir, 'train')

crossvalid_generator = vgg16.generator(crossvalid_dir, 1, shuffle=False)
vgg16.save_bcolz_generator(crossvalid_generator, crossvalid_dir, 'crossvalid')


# In[6]:


train = vgg16.load_bcolz_generator(train_dir, 'train')
validation = vgg16.load_bcolz_generator(crossvalid_dir, 'crossvalid')
vgg16.finetune(train, validation, epochs=2)


# ### Fit the keras model
# 1. Train the updated keras model

# In[7]:


train_dir = TRAIN_DIR
crossvalid_dir = CROSSVALID_DIR

train_generator = vgg16.generator(train_dir, 1, shuffle=False)
vgg16.save_bcolz_generator(train_generator, train_dir, 'train')

crossvalid_generator = vgg16.generator(crossvalid_dir, 1, shuffle=False)
vgg16.save_bcolz_generator(crossvalid_generator, crossvalid_dir, 'crossvalid')


# In[9]:


train = vgg16.load_bcolz_generator(train_dir, 'train')
validation = vgg16.load_bcolz_generator(crossvalid_dir, 'crossvalid')
vgg16.finetune(train, validation, epochs=2)


# ### Save and load the model after couple of epochs

# In[10]:


filename='2_epochs_finetune_0.9684'
vgg16.save_weights(filename)
vgg16.load_weights(filename)


# ## Perform predictions

# In[11]:


preds = vgg16.predict_generator(TEST_DIR)


# In[12]:


preds.shape


# In[13]:


preds[:3]


# In[24]:


from matplotlib import pyplot as plt
import matplotlib.image as mpimg
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# default [6, 4] - controls size of the graph
plt.rcParams['figure.figsize'] = [6, 4]

def plot_images(graph_title, images, figsize=(12,6), titles=None, interp=None, rows=1):
    f = plt.figure(figsize=(12,6))
    f.gca(title=graph_title)
    cols = len(images) // rows if len(images) %2 == 0 else len(images)//rows + 1
    for i in range(len(images)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(images[i], interpolation=None if interp else 'None')
        
# Load the image
img = []
size = 16
for i in range(size):
    if i == 0:
        continue
    file = '%d.jpg' % i
    img_path = TEST_DIR / 'unknown' / file
    img.append(image.load_img(img_path, target_size=(224, 224)))
    
    
# onehot = [0 if pred[0] > pred[1] else 1 for pred in preds[:size] ]    
dog_pred = [pred[1] for pred in preds[:size]]
plot_images('test', img, titles=dog_pred, rows=3)


# In[40]:


# from keras.utils.np_utils import to_categorical
# from sklearn.preprocessing import OneHotEncoder
# def onehot(x): return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())
# preds.shape

# dog_pred = [pred[1] for pred in preds[:size]]
# dog_pred[:3]

import csv
with open('kaggle.csv', 'w') as csvfile:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for idx, pred in enumerate(preds):
        # result = 1 if pred[1] > pred[0] else 0
        result = 1 if pred[0] > pred[1] else 0
        writer.writerow({'id': idx+1, 'label': result})


# ## Kaggle Submit

# ### Prepare csv file and Submit
# 
