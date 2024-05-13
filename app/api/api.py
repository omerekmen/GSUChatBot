from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
import pymongo, datetime, re, string, nltk
from nltk.corpus import stopwords
from collections import Counter


class ChatBot:
    def __init__(self):
        # Gemini
        self.google_api_key="AIzaSyARb5L3obpdE5oxQR-yqcYHKm10oZi8zcc"
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=self.google_api_key)
        # self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.google_api_key)
        self.embeddings = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.parser = StrOutputParser()

        # DB
        self.uri = "mongodb+srv://gsuchatbot:jqaNPBZwJhUbWGZE@chatbot.afy1neg.mongodb.net/?retryWrites=true&w=majority&appName=ChatBot"
        self.public_key = "qtfugelb"
        self.private_key = "ae9ac0bc-ba00-4ae0-87b0-41195eaec07e"

        self.db_cli = pymongo.MongoClient(self.uri)
        self.db = self.db_cli["gsu_chatbot"]
        self.collection = self.db["about_gsu"]

        self.template = """
        Sana verilen context içerisinde soru ve bu soruya uygun çerçevede cevap şu formatta yer almakta:
        'Soru: ...
        Cevap: ...'.
        Verilen cevabı paraphrase ederek güzel bir dille detaylı bir şekilde cevap olarak sun. Soruyu tekrar yazma, sadece cevabı yaz.

        Context: {context}
        """
        self.prompt = PromptTemplate.from_template(self.template)

    def gemini(self, question):
        ai_response = self.model | self.parser
        return ai_response.invoke(question)

    def preprocess_text(self, text, threshold=0.1):
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        
        words = text.split()
        filtered_words = [word for word in words if word not in set(stopwords.words('turkish'))]

        temp_text = ' '.join(filtered_words)
        
        word_freq = Counter(filtered_words)
        total_words = len(filtered_words)
        
        if total_words > 10:
            filtered_text = ' '.join([word for word in filtered_words if word_freq[word] / total_words <= threshold])
        else:
            filtered_text = temp_text
        
        return filtered_text

    def generate_embedding(self, text: str) -> list[float]:
        processed_text = self.preprocess_text(text)
        embedding = self.embeddings.encode(processed_text)
        return embedding.tolist()

    def create_db_embeddings(self, row):
        for doc in self.collection.find({f'{row}': {"$exists": True}}):
            embedding = self.generate_embedding(doc[f'{row}'])
            self.collection.update_one(
                {'_id': doc['_id']},
                {'$set': {f'{row}_embedding': embedding}}
            )

    def insert_doc(self, data):
        for d in data:
            question = d["question"]
            answer = d["answer"]
            tags = d["tags"]

            new_doc = {
                "question": question,
                "answer": answer,
                "tags": tags,
                "date": datetime.datetime.now(),
                "question_embedding": self.generate_embedding(question),
                "answer_embedding": self.generate_embedding(answer),
                "tag_embedding": self.generate_embedding(" ".join(tags)),
            }
            self.collection.insert_one(new_doc).inserted_id

    def delete_doc(self, filter_criteria):
        result = self.collection.delete_one(filter_criteria)
        return f'{result.deleted_count} veri silindi'

    def delete_docs(self, filter_criteria):
        result = self.collection.delete_many(filter_criteria)
        return f'{result.deleted_count} veri silindi'

    def delete_all_docs(self):
        result = self.collection.delete_many({})
        return f'{result.deleted_count} veri silindi'

    def query_search(self, query):
        query_vector = self.generate_embedding(query)
        result_sets = {
            'answer': self.collection.aggregate([
                {
                    "$vectorSearch": {
                        "index": "GSUSearch",
                        "path": "answer_embedding",
                        "queryVector": query_vector,
                        "numCandidates": 100,
                        "limit": 3,
                    }
                },
                {
                    "$project": {
                        "answer": 1,
                        "score": {"$meta": "vectorSearchScore"},
                        "type": {"$literal": "answer"}
                    }
                }
            ]),
            'question': self.collection.aggregate([
                {
                    "$vectorSearch": {
                        "index": "GSUSearch",
                        "path": "question_embedding",
                        "queryVector": query_vector,
                        "numCandidates": 100,
                        "limit": 3,
                    }
                },
                {
                    "$project": {
                        "answer": 1,
                        "score": {"$meta": "vectorSearchScore"},
                        "type": {"$literal": "question"}
                    }
                }
            ]),
            'tag': self.collection.aggregate([
                {
                    "$vectorSearch": {
                        "index": "GSUSearch",
                        "path": "tag_embedding",
                        "queryVector": query_vector,
                        "numCandidates": 100,
                        "limit": 3,
                    }
                },
                {
                    "$project": {
                        "answer": 1,
                        "score": {"$meta": "vectorSearchScore"},
                        "type": {"$literal": "tag"}
                    }
                }
            ])
        }
        combined_results = self.merge_and_rank_results(result_sets)
        return combined_results

    def merge_and_rank_results(self, result_sets):
        combined_scores = {}
        weights = {'answer': 0.15, 'question': 0.85, 'tag': 0.1}

        for result_type, results in result_sets.items():
            for doc in results:
                answer = doc['answer']
                if answer not in combined_scores:
                    combined_scores[answer] = {'score': 0, 'answer': answer}
                combined_scores[answer]['score'] += doc['score'] * weights[result_type]

        sorted_results = sorted(combined_scores.values(), key=lambda x: x['score'], reverse=True)
        return [f"Cevap: {res['answer']}\nScore: {res['score']:.3f}" for res in sorted_results]

    def gsu_chatbot(self, context):
        chain = self.prompt | self.model | self.parser

        response = chain.invoke(
            {
                "context" : context
            }
        )

        return response

if __name__ == "__main__":
    bot = ChatBot()
    query = input("Sorunuzu Giriniz: ")
    results = bot.query_search(query)
    print(bot.gsu_chatbot(results[0]))