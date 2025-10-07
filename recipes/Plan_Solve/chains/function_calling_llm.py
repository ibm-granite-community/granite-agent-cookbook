import os

from langchain.chat_models import init_chat_model

model = "ibm/granite-4-h-small"

model_parameters = {
    "temperature": 0,
    "max_completion_tokens": 2000,
    "repetition_penalty": 1.05,
}


fc_llm = init_chat_model(
    model=model,
    model_provider="ibm",
    url=os.getenv("WATSONX_URL"),
    apikey=os.getenv("WATSONX_APIKEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    params=model_parameters,
)
