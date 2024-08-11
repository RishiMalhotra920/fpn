import sys

from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(color_scheme="Linux", call_pdb=False)
