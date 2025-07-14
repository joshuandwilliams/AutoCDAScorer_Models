import sys
import os
sys.path.append(os.path.expandvars('$HOME/src'))
from post_training import find_and_concatenate_results

find_and_concatenate_results()
