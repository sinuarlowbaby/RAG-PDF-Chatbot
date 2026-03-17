import re


def deduplication(docs , k=20):
    unique_result = []
    seen = set()
    for doc in docs:
        text = doc.page_content.strip()
        text = re.sub(r"\s+", " ", text).strip()
        text = text.lower()
        key = (text, doc.metadata.get("source"))
        if key not in seen:

            unique_result.append(doc)
            seen.add(key)
        if len(unique_result) >= k:
            break
    return unique_result

