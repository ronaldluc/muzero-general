# Use: `from Utils.ipython_autoreload import *` to enable ipython magic autoreload in pycharm
from IPython import get_ipython
ipython = get_ipython()
ipython.magic('%load_ext autoreload')
ipython.magic('%autoreload 2')
ipython.magic('%aimport -pydev_umd')
