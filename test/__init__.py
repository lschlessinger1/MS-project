import os
import sys

top_path = os.path.abspath('..')
if top_path not in sys.path:
    print('Adding to sys.path %s' % top_path)
    sys.path.append(top_path)
