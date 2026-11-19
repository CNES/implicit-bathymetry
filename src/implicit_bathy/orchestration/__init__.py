import os
import sys

ildir, _ = os.path.split(__file__)
if ildir not in sys.path:
    sys.path.append(ildir)
