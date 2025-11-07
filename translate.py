from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

#Create a file named .env and upload your OpeanAI API key.If dont have one create one using this link:"https://platform.openai.com/"  
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

french_prompt = ChatPromptTemplate(
    [
        ("system","You are a Translator assistant.Translate any input language into French language"),
        ("human","Translate this text into French:{text}"),
    ]
)

tamil_prompt = ChatPromptTemplate(
    [
        ("system","You are a Translator assistant.Translate any input language into Tamil language"),
        ("human","Translate this text into Tamil:{text}"),
    ]
)

chain = (RunnableParallel({
        "Tamil":tamil_prompt | model | StrOutputParser(),
        "French":french_prompt | model | StrOutputParser(),
    })
)
print("AI : Hello!,I am a Multi-language Translator model.I translate any language into Tamil and French")
print("AI : Provide Content to be translated")
msgs = []
while True:
    usr_msg = input("You: ")
    if usr_msg=='exit' or usr_msg=='stop':
        break
    msgs.append(HumanMessage(content=usr_msg))
    res = chain.invoke({"text":usr_msg})
    print(f"AI : Tamil - {res['Tamil']}\n     French-{res['French']}\n")
    msgs.append(AIMessage(content=str(res)))
