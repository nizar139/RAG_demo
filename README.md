# Support Chatbot

This is a python project for a RAG system, written mainly using Langchain.

## How Does it work :

### The RAG approach :

Retrieval augmented generation, or RAG, is an architectural approach that can improve the efficacy of large language model (LLM) applications by leveraging custom data. This is done by retrieving data/documents relevant to a question or task and providing them as context for the LLM.


![image](https://github.com/nizar139/support-chatbot/assets/93913464/03b6b43a-fa43-4810-9c9b-4de5df2cebae)


In our context, the RAG apprach can enable us to answer questions on specialized knowledge

### The vector Database :

![image](https://github.com/nizar139/support-chatbot/assets/93913464/5dc21c5a-d579-47d2-bc05-379402a90b13)


We use a Chroma vector database (stored in `database` directory), created using the script `create_vector_db.py`, where chunks are stored as Embeddings (vectors). For that purpose we use HuggingFace embeddings, which are easy to access and use. vector databases enable us to use similarity search to find relevant knowledge to our prompts.

We indexed several markdown files in the `docs` directory using langchain splitters (first by header then to smaller chunks with overlap if necessary)

### The prompting :

In order to retrieve the relevant context, we query the vector database by calculating a similarity score between the question and the stored knowledge.

the main prompt is divised in two parts :
- A system prompt that describe the use case and provides the necessay context.
- The Question

### The LLM :

The system is implemented currently to use LLMs available with HuggingFace transformers locally or by inference endpoint. the LLM repo can be specified in `src/config.py`

### Future Improvments :

There are several ideas to improve this RAG system :

- Updgrade the indexing and retrieval process.
- Maintain a conversation history for a chatbot system.
- Allow the LLM to use the retrieval multiple times (as a tool) for chain-of-thoughts answers.
- Add a feedback loop for the user to evaluate the quality of the responses.
- Allow the user to view and modify the retrieved context.
- Make the code asynchronious.

## How to run the files

### Preparations

This project uses python 3.10. In order to run the code, you need first to create a virtual environment and install the required packages via :

```
pip install -r requirements.txt
```

Before being able to run the code, you need to either have a [HUGGINGFACE API TOKEN](https://huggingface.co/settings/tokens).

You can either add it to your environment variables, using the name `HUGGINFACEHUB_API_TOKEN`
or you can put them in config.py in the designated place, or a .env file.

```                          
HUGGINFACEHUB_API_TOKEN = os.getenv("HUGGINFACEHUB_API_TOKEN", "your token") 
```

For each provider, you need to change `LLM_PROVIDER` in `config.py` accordinly 
```
providers = ["huggingface_endpoint", "huggingface_local"]
LLM_PROVIDER = "your provider from the list"
``` 

### run the streamlit app 

in a terminal where the env is activated, make sure that you are in the root directory of the project, some directory names variables won't work otherwise, then :
```
streamlit run front.py
```
to stop the app you need to press `ctrl+c` in the used terminal

### run a console chatbot

in a terminal where the env is activated, make sure that you are in the root directory of the project, some directory names variables won't work otherwise, then :
```
python -m src.backend
```
to stop the app you can either input `q` or `quit`.

### run an offline inference on multiple questions

in a terminal where the env is activated, make sure that you are in the root directory of the project, some directory names variables won't work otherwise, then :
```
python -m src.make_responses --input_path=/path/to/input/json --output_path=/path/to/output/json
```
to stop the app you can either input `q` or `quit`.

### Update the vector database 

This project uses a Chroma vector database that includes relevant knowledge from the documents. the vector database is in the directory `database`

the script `create_vector_db.py` takes all the Markdown files in the `docs` directory, and recreates a vector database using the knowledge in these documents.
in order to run it, make sure you placed the files you want to be included in `docs` directory.

in a terminal where the env is activated, make sure that you are in the root directory of the project, some directory names variables won't work otherwise, then :
```
python -m create_vector_db.py --docs_path=docs/ --save_path=database/
```

