from pathlib import Path

from ibm_granite_community.notebook_utils import get_env_var
from langchain_core.utils.utils import convert_to_secret_str

ROOT_DIR = Path(__file__).resolve().parent.parent
AV_STOCK_API_KEY = convert_to_secret_str(get_env_var("AV_STOCK_API_KEY", "unset"))
WEATHER_API_KEY = convert_to_secret_str(get_env_var("WEATHER_API_KEY", "unset"))
