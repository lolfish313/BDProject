from langchain_openai import ChatOpenAI


class ChatOpenAIModel:

    @classmethod
    def chatOpenAI(cls,
                   model,
                   base_url,
                   api_key="EMPTY",
                   temperature=0.01,
                   # top_p=0.9,
                   **kwargs):
        llm = ChatOpenAI(model=model,
                         base_url=base_url,
                         api_key=api_key,
                         extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                         temperature=temperature,
                         # top_p=top_p,
                         **kwargs)
        return llm