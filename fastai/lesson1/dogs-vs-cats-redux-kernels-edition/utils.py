# Common imports
import itertools
import os

import numpy as np
import numpy.random as rnd

# to make this notebook's output stable across runs
rnd.seed(42)

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import seaborn as sns
sns.set()

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# default [6, 4] - controls size of the graph
plt.rcParams['figure.figsize'] = [6, 4]

def print_txt_results(results, filenames):
    for idx, result in enumerate(results):
        print('\n#### Contents of "{}"'.format(filenames[idx]))
        for text in result:
            try:
                print(text.decode('utf8'), end='')
            except:
                # oops...not a byte string
                print(text, end='')


def plot_images(graph_title, images, figsize=(12, 6), titles=None, interp=None, rows=1):
    f = plt.figure(figsize=(12, 6))
    f.gca(title=graph_title)
    cols = len(images) // rows if len(images) % 2 == 0 else len(images) // rows + 1
    for i in range(len(images)):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=10)
        plt.imshow(images[i], interpolation=None if interp else 'None')
    plt.show()


def list_files(filepath, num_of_files=5, num_of_lines=30, inspect=True, images=False):
    import tarfile

    filepath = os.getcwd() + filepath
    with tarfile.open(filepath) as tar:
        all_members = tar.getmembers()

        files_found = sum(1 for member in all_members if member.isfile())
        print("Number of files in {}: {}".format(filepath, files_found))

        partial_members = [member for member in tar.getmembers() if member.isfile()][0:num_of_files]
        print([member.name for member in partial_members])
        if inspect:
            filenames = []
            result = []

            for member in partial_members:
                with tar.extractfile(member) as content:
                    filenames.append(member.name)
                    if images is True:
                        result.append(mpimg.imread(content, format='PNG'))
                    else:
                        result.append(content.readlines()[0:num_of_lines])

            return result, filenames

#             if images is True:
#                 plot_images(os.path.basename(filepath), result, filenames)
#             else:
#                 [print(text.decode('utf8'), end='', flush=True) for text in result]

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
