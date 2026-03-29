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
        # Use PyPDFLoader directly for individual files
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    return documents

