# Agent TDD — Test-Driven Development for Agents

This recipe demonstrates applying Test-Driven Development (TDD) to function-calling agents. It includes a short demo notebook and minimal environment requirements.

- Notebook: [test_driven_agent_demo.ipynb](Function_Calling_Agent_TDD.ipynb)

[![Open In Colab](https://colab.research.google.com/drive/1bGr4TEiebbf6tZqC9hhpwIpYJGd2Bdo-?usp=sharing)]

Quick start

1. Create and activate a fresh virtual environment (Python 3.10+ recommended).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Open the notebook in Jupyter or click the Colab button above.

Notes
- The notebook expects the helper utils from the IBM Granite community package. The `requirements.txt` includes a VCS reference to install it.
- Ensure the first notebook cell includes the line shown in the recipe guide to access secrets via `get_env_var` when needed:

```python
# %pip install git+https://github.com/ibm-granite-community/utils
```
