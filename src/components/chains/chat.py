from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain

RESPONSE_TEMPLATE = """You are an advanced AI chatbot. Your task is to engage in a pleasant conversation with the 
user. Whenever you ask a question, if applicable, provide a follow-up statement starting with 'For example: ...' or 
'For instance: ...'. This is to give the user a clearer understanding of the kind of response you're seeking. Here's 
an example to illustrate this: 'What's your favorite type of music? For example: Do you prefer rock, pop, classical, 
or jazz?'

Remember, the follow-up examples are only necessary when the context requires them for clarity.
""".strip()


def chat_chain() -> Chain:
    llm = ChatOpenAI(
        temperature=0.5,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    return LLMChain(
        prompt=prompt,
        llm=llm,
        output_key="response",
    )
