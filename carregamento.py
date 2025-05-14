from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders.pdf import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

caminho = 'arquivos/TSP_CMC_54360.pdf'
loader = PyPDFLoader(caminho)
documentos = loader.load()


llm = ChatOpenAI(model="gpt-4.1-2025-04-14")
chain = load_qa_chain(llm, chain_type="stuff")

print("Chain loaded successfully.")


pergunta = 'Quais assuntos são tratados no documento?'

chain.run(input_documents=documentos[:10], question=pergunta)
