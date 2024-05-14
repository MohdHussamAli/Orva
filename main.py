import json
from flask import Flask, request, jsonify
import string
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import json
import os
import numpy as np
import pandas as pd
from flask import send_file
from flask import Flask, render_template
import ast
import re
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)

project="qdmeds"
location="us-central1"
index_name= "orva_poc_index_deployed_05030849"
df=pd.read_csv("./Orva_Embed_File.csv")
print("df length: ", len(df))

df_input=pd.read_csv("./orva_data.csv")
# print(df_input)
print("df_input: ", len(df_input))

aiplatform.init(project=project,location=location)
vertexai.init()
model = GenerativeModel("gemini-1.5-pro-preview-0409")
orva_poc_index_ep = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name='3989120544548061184')

no_match_response ="""Unfortunately, I don't have enough information to answer your specific question accurately but we're always working to expand my capabilities. However, I can help you explore resources on managing pain, soreness, swelling, and incision site care after surgery. Just let me know which topic you'd like to learn more about!

**Tip**: I'm best equipped to answer questions when they are asked as complete sentences."""

def generate_text_embeddings(sentences) -> list:    
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    embeddings = model.get_embeddings(sentences)
    vectors = [embedding.values for embedding in embeddings]
    return vectors

def generate_context(data,text_column='title'):
    concatenated_sentences = ''
    for _, row in data.iterrows():
        concatenated_sentences += row['title'] + "\n" 
    return concatenated_sentences.strip()

def get_vector(data):
    concatenated_sentences = []
    for _, row in data.iterrows():
        concatenated_sentences.append(row['embedding'])
    return concatenated_sentences[0]

def extract_empathetic_line(text):
    pattern = r'\**\s*[Ee]mpathetic line\s*[:*]*\s*(.*?)(?=\.)'
    match = re.search(pattern, text)
    if match:
        empathetic_line = match.group(1).strip()
        return empathetic_line
    else:
        return ""

@app.route('/dialogflow', methods=['GET', 'POST'])
def dialogflow():
    data = request.get_json()
    print("data=>",json.dumps(data))
    query = data["text"]
    return find_match_response(query)[1]

def find_match_response(query):
    print("Query:", query)
    context = ""
    query=[query]
    qry_emb=generate_text_embeddings(query)
    response = orva_poc_index_ep.find_neighbors(
        deployed_index_id = index_name,
        queries = [qry_emb[0]],
        num_neighbors = 5,approx_num_neighbors=10,
        return_full_datapoint=True
    )
    matching_ids = []
    for idx, neighbor in enumerate(response[0]):
        matching_ids.append(neighbor.id)
        id = np.int64(neighbor.id)
        similar = df.query("id == @id", engine="python")
        context += generate_context(similar) + "\n"

    print("==> matching ids: ", matching_ids)
    prompt=f"""You are a triage nurse assistant to answer user queries post knee surgery. Based on the context provided, answer the question verbatim from the context. context: {context}, query:{query}
    Follow these guidelines:
    + Answer the Human's query and make sure you mention exact details from
    the context, using exactly the same words as the context if possible.
    + The answer must be based only on the context and not introduce any words or additional
    information.
    + All numbers, like price, date, time or phone numbers must appear exactly as
    they are in the context.
    + The answer MUST be in English.
    + Do not make up any words or the answer: If the answer cannot be found in the context,
    you admit that you don't know and you answer NOT_ENOUGH_INFORMATION.
    """
    chat = model.start_chat(history=[])
    result = chat.send_message(prompt)
    vertext_text = result.text
    vertex_response=result.text.strip()
    print("Vertex Response: ", vertex_response)
    # reverse lookup based on the response:
    res_emb=generate_text_embeddings([vertex_response])
    input_ranks =[]
    best_input_id = 0
    best_input_rank =0
    for matching_id in matching_ids:
        input_emb = get_vector(df.query(f"id == {matching_id}", engine="python"))
        input_vec = ast.literal_eval(input_emb)
        similarity_rank = cosine_similarity([res_emb[0]], [input_vec])[0][0]
        if(similarity_rank>best_input_rank):
            best_input_rank = similarity_rank
            best_input_id = matching_id
        input_ranks.append(similarity_rank)
        
    response_text = ''
    answer_id =''
    if(vertex_response == "NOT_ENOUGH_INFORMATION"):
        answer_id =0
        response_text = no_match_response
    else:
        answer_id = best_input_id
        response_text = generate_context(df_input.query(f"id == {answer_id}", engine="python"),text_column='title')
    
    print("Answer :",answer_id)
    
    json_response= jsonify(
        {
            'fulfillment_response': {
                'messages': [
                    {
                        'text': {
                            'text': [f"{response_text}"]
                        }
                    }
                ]
            }
        }
    )
    return [answer_id,json_response]



def test_queries():
    df=pd.read_csv("./test.csv")
    df = df.reset_index()  # make sure indexes pair with number of rows
    response_row =[]
    for index, row in df.iterrows():
        response = find_match_response(row['query'])
        response_row.append([index,row['query'],row['expected_id'],response[0],(str(row['expected_id'])==response[0])])
    print(response_row)
    
if __name__ == "__main__":
    with app.app_context():
        # test_queries()
        # find_match_response("Can i plant a garden") #104 -f
        # find_match_response("how should i go upstairs") #104 -f
        #find_match_response("my knee doesnt feel natural at all") #248 #252 - p
        #find_match_response("my knee itches") #356 -f
        # find_match_response("my physical therepy hurts too much to go to") #61 -p
        # find_match_response("what do i need to know about gardening")
        # find_match_response("what do i need to know about gardening")
        # find_match_response("where do you land a rover") #none -p
        # find_match_response("This shouldnt do any work at all") #none -
        app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))