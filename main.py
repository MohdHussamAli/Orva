import json
from flask import Flask, request, jsonify

app = Flask(__name__)


#import uvicorn
import string
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import json
import os
import numpy as np
import pandas as pd

@app.route('/', methods=['GET', 'POST'])
def home():
    return 'OK', 200

project="qdmeds"
location="us-central1"
index_name="orva_poc_index_deployed_05030849"
df=pd.read_csv("./Orva_Embed_File.csv")
df

aiplatform.init(project=project,location=location)
vertexai.init()
model = GenerativeModel("gemini-1.5-pro-preview-0409")
# model = GenerativeModel("gemini-pro")
orva_poc_index_ep = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name='3989120544548061184')

def generate_text_embeddings(sentences) -> list:    
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    embeddings = model.get_embeddings(sentences)
    vectors = [embedding.values for embedding in embeddings]
    return vectors

def generate_context(data):
    concatenated_sentences = ''
    for _, row in data.iterrows():
        concatenated_sentences += row['title'] + "\n" 
    return concatenated_sentences.strip()


@app.route('/dialogflow', methods=['GET', 'POST'])
def dialogflow():

    data = request.get_json()
    #print(data)
    print("data=>",json.dumps(data))
    #query = query_request.get_json()
    #print("Query:", query)
    #query = query_request.query
    query = data["text"]
    print("Query:", query)
    context = ""
    #query=["second knee replacement"]
    #query=["second knee replacement "]
    query=[query]
    print("list", query)
    qry_emb=generate_text_embeddings(query)
    response = orva_poc_index_ep.find_neighbors(
        deployed_index_id = index_name,
        queries = [qry_emb[0]],
        num_neighbors = 10
    )

    for idx, neighbor in enumerate(response[0]):
        id = np.int64(neighbor.id)
        similar = df.query("id == @id", engine="python")
        context += generate_context(similar) + "\n"

    #print("Context:", context)
    prompt=f"""Based on the context provided, answer the question appropriately. context: {context}, query:{query}
    Follow these guidelines:
    + Answer the Human's query and make sure you mention all relevant details from
    the context, using exactly the same words as the cotext if possible.
    + The answer must be based only on the context and not introduce any additional
    information.
    + All numbers, like price, date, time or phone numbers must appear exactly as
    they are in the context.
    + Give as comprehensive answer as possible given the context. Include all
    important details, and any caveats and conditions that apply.
    + The answer MUST be in English.
    + Don't try to make up an answer: If the answer cannot be found in the context,
    you admit that you don't know and you answer NOT_ENOUGH_INFORMATION.
    You will be given a few examples before you begin.
    Begin! Let's work this out step by step to be sure we have the right answer."""


    chat = model.start_chat(history=[])
    result = chat.send_message(prompt)
    print("Vertex Response:",result.text)
    vertex_response=result.text
    #return result.text
    return jsonify(
        {
            'fulfillment_response': {
                'messages': [
                    {
                        'text': {
                            'text': [vertex_response]
                        }
                    }
                ]
            }
        }
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
