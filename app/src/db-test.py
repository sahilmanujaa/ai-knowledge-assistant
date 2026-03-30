import chromadb

client = chromadb.Client()

collection = client.create_collection(name="my_docs")

collection.add(
    documents=["GenAI is amazing"],
    ids=["1"]
)

results = collection.query(
    query_texts=["What is GenAI?"],
    n_results=1
)

print(results)
