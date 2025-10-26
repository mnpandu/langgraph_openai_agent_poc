import os
from io import StringIO
from contextlib import redirect_stdout
from langchain_core.tools import tool
import pandas as pd

@tool
def exec_python_code(code: str) -> str:
    """Execute Python code safely and return stdout."""
    buffer = StringIO()
    try:
        safe_globals = {
            'pd': pd,
            'plt': __import__('matplotlib.pyplot'),
            'np': __import__('numpy'),
            'os': os,
        }
        with redirect_stdout(buffer):
            exec(code, safe_globals)
    except Exception as e:
        return f"Error: {e}"
    return buffer.getvalue().strip() or "(No output)"
