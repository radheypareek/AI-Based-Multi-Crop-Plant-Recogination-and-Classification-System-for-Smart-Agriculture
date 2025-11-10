# submission/app.py - Root Streamlit entrypoint wrapper
import sys, os, runpy
# Ensure project root is on sys.path so `import src.*` works
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# Delegate to the actual Streamlit app module in src/
runpy.run_module('src.streamlit_app', run_name='__main__')
