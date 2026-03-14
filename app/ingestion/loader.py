import os
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

def doc_loader():
    base_path = os.path.dirname(__file__)
    project_root = os.path.join(base_path, "../")
    data_path = os.path.join(project_root, "data")

    loader = DirectoryLoader(
        data_path,
         glob="*.pdf",
         loader_cls=PyPDFLoader,
         use_multithreading=True
         )

    documents = loader.load()
    return documents

