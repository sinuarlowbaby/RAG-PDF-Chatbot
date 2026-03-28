import os
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

    

def load_documents(files):
    base_path = os.path.dirname(__file__)
    project_root = os.path.join(base_path, "../")
    data_path = os.path.join(project_root, "data")
    documents = []
    for file_path in files:
        if not file_path.endswith(".pdf"):
            continue
        loader = DirectoryLoader(
                file_path,
                glob="*.pdf",
                loader_cls=PyPDFLoader,
                use_multithreading=True
            )
        documents.extend(loader.load())
    return documents

