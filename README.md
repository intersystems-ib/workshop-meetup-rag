# workshop-llm
Workshop to create a RAG application using LLM models. 

This workshop is developed in Python 🐍 (Jupyter Notebook) and InterSystems IRIS.

The main purpose is to show you the main steps to create a RAG application using an LLM and a vector database.

You can find more in-depth information in https://learning.intersystems.com.

# What do you need to install? 
* [Git](https://git-scm.com/downloads) 
* [Docker](https://www.docker.com/products/docker-desktop) (if you are using Windows, make sure you set your Docker installation to use "Linux containers").
* [Docker Compose](https://docs.docker.com/compose/install/)
* [Visual Studio Code](https://code.visualstudio.com/download) + [InterSystems ObjectScript VSCode Extension](https://marketplace.visualstudio.com/items?itemName=daimor.vscode-objectscript)

# Setup
Build the image we will use during the workshop:

Clone the repository:
```bash
git clone https://github.com/intersystems-ib/workshop-llm
cd workshop-llm
```

Build the image:
```bash
docker compose build
```

Run the containers:
```bash
docker compose up -d
```

After running the containers, you should be able to access to:
* InterSystems IRIS [Management Portal](http://localhost:52774/csp/sys/UtilHome.csp). You can login using `superuser` / `SYS`
* [Jupyter Notebook](http://localhost:8888) 

# Explore RAG applications using Jupyter

## Medicine Leaflet examples

You have some medicine leaflets (in spanish) in [./data](./data).

This example is about creating a RAG Q&A application that can answer questions about those medicine leaflets.

Open [Jupyter Notebook](http://localhost:8888), there you can find:
* [QA-PDF-LLM.ipynb](./jupyter/QA-PDF-LLM.ipynb) - RAG example using [MistralAI](https://mistral.ai) LLM 
* [QA-PDF-Local.ipynb](./jupyter/QA-PDF-Local.ipynb) - RAG example using a local LLM

![alt text](/images/jupyter.png)

You can test the project step by step or execute it at one time, feel free.

## Hoolefoods data model text to SQL

This example is about a company called Holefoods that sells food with some hole on it :)

Using the sales data model of the company, the goal is to create an assistant that can translate natural language questions into valid SQL that answer the question.

In [Jupyter Notebook](http://localhost:8888), you will find:
* [QA-SQL-LLM.ipynb](./jupyter/QA-SQL-LLM.ipynb) - text to SQL example using OpenAI LLM.

# Create other applications

There are some other examples you can try to build and modify in your local environment.

First of all, create a new environment and install some requirements:

```bash
# create a local venv environment
python3 -m venv .venv

# activate venv
source .venv/bin/activate

# install dependencies
pip3 install -r requirements.txt
```

Create an `.env` file for storing API keys for OpenAI / MistralAI. They will be used in the applications.

```
OPENAI_API_KEY="your-api"
MISTRAL_API_KEY="your-api"
```

## Text to SQL service API 
You can find a sample Text to SQL based on [QA-SQL-LLM.ipynb](./jupyter/QA-SQL-LLM.ipynb) [here](python/holefoods_text2sql/main.py).

You can run it like this:

```bash
cd python/holefoods_text2sql
fastapi dev main.py
```

Then open http://127.0.0.1:8000/docs to explore the API and try it out using the web client.

## Streamlit Assistant
There is also a great example of a langchain / streamlit chatbot assitant in https://alejandro-ao.com/how-to-use-streaming-in-langchain-and-streamlit/

You can play with it here as well:

```bash
cd python/assitant
streamlit run chatbot.py
```

Then open http://localhost:8501 and have a look at it.

Are you able to add the logic to reproduce the Medicine Leaflet example in the assitant ?
