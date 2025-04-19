# Dynamically import all test_*.py modules so "python -m unittest tests" discovers them
import os, glob

here = os.path.dirname(__file__)
for path in glob.glob(os.path.join(here, "test_*.py")):
    mod = os.path.splitext(os.path.basename(path))[0]
    __import__(f"{__name__}.{mod}", fromlist=["*"])
