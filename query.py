# https://www.mongodb.com/docs/atlas/atlas-search/knn-beta/
import pymongo
from openai.embeddings_utils import get_embedding
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

MONGO_URI = "<MONGO_URI>"
MONGODB_DATABASE="llamatest"
QUESTION = "What steps should banks take to modernize their business?"

def main():
    vectorizedQuery = get_embedding(QUESTION,engine="text-search-ada-query-001")
    # Connect to the MongoDB server
    client = pymongo.MongoClient(MONGO_URI)
    # Get the collection
    collection = client.llamatest.vectorific
    pipeline = [
        {
            "$search": {
                "knnBeta": {
                    "vector": vectorizedQuery,
                    "path": "embeddings",
                    "k": 5
                }
            }
        },
        {
            "$project": {"embeddings":0}
        },
        {
            "$limit": 5 #lets assume the first 3 chunks is the most useful
        }
    ]
    results = collection.aggregate(pipeline)

    # Use the vector-search results to provide an enhanced context for the AI model
    context = "Use the context below to answer the question further below.\n\nCONTEXT: "
    for i,result in enumerate(results):
        context += "\n"+str(i)+") "+result["content"]+"\n"
    
    # set llm temperature
    llm = OpenAI(temperature=.2)
    # set llm template, preceded by context we just created with vector search
    template = context+"""\n USE THE CONTEXT TO ANSWER THIS QUESTION. DO NOT INCLUDE INSTRUCTIONS OR CONTEXT IN RESPONSE:
    Question: {text} \n
    Answer:\n\n\n
    """
    prompt_template1 = PromptTemplate(input_variables=["text"], template=template)
    prompt_template2 = PromptTemplate(input_variables=["text"], template="Q:{text}\nA:")
    answer_chain_enhanced = LLMChain(llm=llm, prompt=prompt_template1)
    answer_chain_reg = LLMChain(llm=llm, prompt=prompt_template2)
    print("\nQUESTION:"+QUESTION+"\n\n ANSWER:\n")
    answer = answer_chain_enhanced.run(QUESTION)
    answer2 = answer_chain_reg.run(QUESTION)
    print(answer)
    print("\n\n\nENHANCED CONTEXT \n")
    print(context)
    print("\n\n\n\nANSWER WITHOUT ENHANCED CONTEXT:\n")
    print(answer2)
main()