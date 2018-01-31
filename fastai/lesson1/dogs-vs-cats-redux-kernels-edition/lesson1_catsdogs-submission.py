
# coding: utf-8

# ## Preamble

# In[121]:


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
get_ipython().run_line_magic('aimport', 'utils')
from utils import *
from keras_vgg16 import *


# ## Model preparation

# ### Finetune the keras model
# * Pop the last layer, freeze all layers, add a softmax layer and update set of classes

# In[2]:


vgg16 = KerasVgg16(WEIGHTS_DIR)
vgg16.create_model(learning_rate = 0.01, ttl_outputs=2)


# In[3]:


train_dir = SAMPLE_TRAIN_DIR
crossvalid_dir = SAMPLE_CROSSVALID_DIR

train_generator = vgg16.generator(train_dir, 1, shuffle=False)
vgg16.save_bcolz_generator(train_generator, train_dir, 'train')

crossvalid_generator = vgg16.generator(crossvalid_dir, 1, shuffle=False)
vgg16.save_bcolz_generator(crossvalid_generator, crossvalid_dir, 'crossvalid')


# In[4]:


train = vgg16.load_bcolz_generator(train_dir, 'train')
validation = vgg16.load_bcolz_generator(crossvalid_dir, 'crossvalid')
#vgg16.finetune(train, validation, epochs=2)


# ### Fit the keras model
# 1. Train the updated keras model

# In[5]:


train_dir = TRAIN_DIR
crossvalid_dir = CROSSVALID_DIR

train_generator = vgg16.generator(train_dir, 1, shuffle=False)
vgg16.save_bcolz_generator(train_generator, train_dir, 'train')

crossvalid_generator = vgg16.generator(crossvalid_dir, 1, shuffle=False)
vgg16.save_bcolz_generator(crossvalid_generator, crossvalid_dir, 'crossvalid')


# In[6]:


train = vgg16.load_bcolz_generator(train_dir, 'train')
validation = vgg16.load_bcolz_generator(crossvalid_dir, 'crossvalid')
#vgg16.finetune(train, validation, epochs=2)


# ### Save and load the model after couple of epochs

# In[7]:


filename='2_epochs_finetune_0.9684'
#vgg16.save_weights(filename)
vgg16.load_weights(filename)


# ## CrossValidation

# In[8]:


vgg16.classifications(crossvalid_generator)


# Cat is 0 and Dog is 1

# In[103]:


crossvalid_dir = SAMPLE_CROSSVALID_DIR

crossvalid_generator = vgg16.generator(crossvalid_dir, 1, shuffle=False)

batch_size = 10
validation = vgg16.load_bcolz_generator(crossvalid_dir, 'crossvalid', batch_size=batch_size, shuffle=False)
print(validation.X.shape)

#test_batch, preds = vgg16.predict_generator(TEST_DIR)
validation_preds = vgg16.model.predict_generator(validation, steps=validation.samples / batch_size)
validation_preds.shape


# In[10]:


vgg16.save_array('validation_preds_' + filename, validation_preds)
preds = vgg16.load_array('validation_preds_' + filename)
preds.shape


# ## Perform predictions

# ### Confusion Matrix

# In[144]:


expected=np.array([1.0 if valid[1] == 1 else 0.0 for valid in validation.y])
print("Expected Beginning %s" % expected[:5])
print("Expected End %s" % expected[-5:])

#Round our predictions to 0/1 to generate labels
our_predictions = preds[:,0]
actual = np.round(1-our_predictions)
print("Actual %s" % actual)

print(crossvalid_dir)
print(validation.X[0].shape)
print(len(crossvalid_generator.filenames))


# In[139]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
from utils import *

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(expected, actual)
plot_confusion_matrix(cm, crossvalid_generator.class_indices, normalize=False, title='Confusion matrix')


# ### Plot sample predictions

# In[179]:


get_ipython().run_line_magic('reload_ext', 'autoreload')

# correct = np.where(actual==expected)[0]
# print ("Found %d correct labels" % len(correct))
# np.random.shuffle(correct)
# correct = correct[:5]
# images = [image.load_img(Path(crossvalid_dir) / crossvalid_generator.filenames[c], target_size=(224, 224)) for c in correct]
# titles = [actual[c] for c in correct]
# plot_images('True Positive', images, rows=1, titles=titles)

def plot(title, criteria):
    matches = np.where(criteria)[0]
    print('Found %d %s labels' % (len(matches), title))
    np.random.shuffle(matches)
    matches = matches[:5]
    images = [image.load_img(Path(crossvalid_dir) / crossvalid_generator.filenames[m], 
                             target_size=(224, 224)) for m in matches]
    titles = ['%d - %s' % (actual[m], crossvalid_generator.filenames[m].split('/')[1]) for m in matches]
    print('Found in %s' % crossvalid_dir)
    plot_images(title, images, rows=1, titles=titles)


# In[182]:


plot('True Positive', (expected == 1) & (actual == expected))


# In[172]:


plot('True Negative', (expected == 0) & (actual == expected))


# In[183]:


plot('False Positive', (expected == 0) & (actual == 1))


# In[184]:


plot('False Negative', (expected == 1) & (actual == 0))


# ## Predictions

# In[222]:


test_dir = TEST_DIR

print('Loading %s' % test_dir)
test_generator = vgg16.generator(test_dir, 1, shuffle=False)

batch_size = 10
vgg16.save_bcolz_generator(test_generator, test_dir, 'test')
test = vgg16.load_bcolz_generator(test_dir, 'test', batch_size=batch_size, shuffle=False)
print(test.X.shape)


# In[226]:


print(test.samples)
print(test.batch_size)

# OOME
# test_preds = vgg16.model.predict_on_batch(test.X)

test_preds = vgg16.model.predict_generator(test, steps=(test.samples / test.batch_size) + 1, verbose=1)
test_preds.shape


# In[234]:


vgg16.save_array('preds_' + filename, test_preds)


# In[237]:


preds = vgg16.load_array('preds_2_epochs_finetune_0.9684')


# In[238]:


preds.shape


# In[240]:


#Grab the dog prediction column
isdog = preds[:,1]
print ("Raw Predictions: " + str(isdog[:5]))
print ("Mid Predictions: " + str(isdog[(isdog < .6) & (isdog > .4)]))
print ("Edge Predictions: " + str(isdog[(isdog == 1) | (isdog == 0)]))


# In[244]:


#Swap all ones with .95 and all zeros with .05
isdog = isdog.clip(min=0.05, max=0.95)
#Extract imageIds from the filenames in our test/unknown directory 
filenames = test_generator.filenames
ids = np.array([int(f[8:f.find('.')]) for f in filenames])
ids.shape


# In[245]:


subm = np.stack([ids,isdog], axis=1)
subm[:5]


# In[247]:


submission_file_name = 'submission1.csv'
np.savetxt(submission_file_name, subm, fmt='%d,%.5f', header='id,label', comments='')


# In[263]:


from IPython.display import FileLink
FileLink(str(current_dir) +  "/" + submission_file_name)

