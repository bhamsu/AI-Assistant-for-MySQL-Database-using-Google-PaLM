import os
import sqlalchemy
from dotenv import load_dotenv
from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from fewShots import few_shots
from mysql_prompt import mysql_prompt


class Database_Assistant:

    def __init__(self):

        self.few_shot_prompt = None
        load_dotenv()
        self.llm = GooglePalm(google_api_key = os.environ["GOOGLE_API_KEY"], temperature = 0.1)
        # poem = llm("write a poem for the love of my life")
        # print(poem)
        print("Google PaLM API verified and connected with the PaLM server...")

        self.db_user = os.environ["DB_USER"]
        self.db_password = os.environ["DB_PASSWORD"]
        self.db_host = "localhost"
        self.db_name = "atliq_tshirts"

        self.example_selector = None
        self.db = None

    def mysql_connection(self):

        try:
            self.db = SQLDatabase.from_uri(
                f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}",
                sample_rows_in_table_info = 3)
            # print(db.table_info)
            print("MySQL Connection Successful...")
        except sqlalchemy.exc.OperationalError:
            print("MySQL Database Login credentials invalid...")

    def text_embedding(self):

        db_chain = SQLDatabaseChain.from_llm(self.llm, self.db, verbose = True)
        # qns1 = db_chain.run("How many t-shirts do we have left for nike in extra small size and white color?")
        # print(qns1)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        # embeddings = INSTRUCTOR('hkunlp/instructor-large')
        print("Text Embedding model loaded from HuggingFace...")

        to_vectorize = [" ".join(example.values()) for example in few_shots]
        vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas = few_shots)
        self.example_selector = SemanticSimilarityExampleSelector(vectorstore = vectorstore, k = 2)
        self.example_selector.select_examples({"Question": "How many Adidas T shirts I have left in my store?"})
        # print(PROMPT_SUFFIX)

    def chain_creation(self):

        example_prompt = PromptTemplate(
            input_variables=["Question", "SQLQuery", "SQLResult", "Answer", ],
            template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
        )
        # print(_mysql_prompt)

        self.few_shot_prompt = FewShotPromptTemplate(
            example_selector = self.example_selector,
            example_prompt = example_prompt,
            prefix = mysql_prompt,
            suffix = PROMPT_SUFFIX,
            input_variables = ["input", "table_info", "top_k"],  # These variables are used in the prefix and suffix
        )

    def run(self, text):

        new_chain = SQLDatabaseChain.from_llm(self.llm, self.db, verbose = True, prompt = self.few_shot_prompt)
        # response = new_chain("How many white color Levi's shirt I have?")
        response = new_chain.run(text)
        # print(response)
        return response

    def __call__(self, *args, **kwargs):

        self.mysql_connection()
        self.text_embedding()
        self.chain_creation()
        return self.run(args[0])

    def __del__(self):
        pass