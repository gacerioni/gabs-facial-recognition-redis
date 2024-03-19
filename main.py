import os

import numpy as np
import redis
from deepface import DeepFace
from dotenv import load_dotenv
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from config.logger_config import setup_logger

# Load environment variables
load_dotenv()

# Setup logger
logger = setup_logger()

# Initialize global variables
REDIS_URL = os.getenv('GABS_REDIS_URL', "redis://localhost:6379")
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "Dlib", "ArcFace"]
EMBEDDING_MODEL = models[0]
ORL_DATA_PATH = "data/orl/"

# Get a Redis connection
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
print(r.ping())


def store_models():
    for person in range(1, 41):
        person = "s" + str(person)
        r.json().set(f"face:{person}", "$", {'person_id': person})
        r.json().set(f"face:{person}", "$.embeddings", [])
        for face in range(1, 6):
            facepath = ORL_DATA_PATH + person + "/" + str(face) + '.bmp'
            print("Training face: " + facepath)
            vec = DeepFace.represent(facepath, model_name=EMBEDDING_MODEL, enforce_detection=False)[0]['embedding']
            embedding = np.array(vec, dtype=np.float32).astype(np.float32).tolist()
            r.json().arrappend(f"face:{person}", '$.embeddings', embedding)


def create_index():
    indexes = r.execute_command("FT._LIST")
    if "face_idx" not in indexes:
        index_def = IndexDefinition(prefix=["face:"], index_type=IndexType.JSON)
        schema = (VectorField("$.embeddings[*]", "HNSW", {"TYPE": "FLOAT32", "DIM": 2622, "DISTANCE_METRIC": "COSINE"},
                              as_name="embeddings"))
        r.ft('face_idx').create_index(schema, definition=index_def)
        print("The index has been created")
    else:
        print("The index exists")


def find_face(facepath):
    vec = DeepFace.represent(facepath, model_name=EMBEDDING_MODEL, enforce_detection=False)[0]['embedding']
    embedding = np.array(vec, dtype=np.float32).astype(np.float32).tobytes()
    q = Query("*=>[KNN 1 @embeddings $vec AS score]").return_field("score").dialect(2)
    res = r.ft("face_idx").search(q, query_params={"vec": embedding})

    for face in res.docs:
        print(face.id.split(":")[1])
        return face.id.split(":")[1]


def test():
    success = 0
    for person in range(1, 41):
        person = "s" + str(person)
        for face in range(6, 11):
            facepath = ORL_DATA_PATH + person + "/" + str(face) + '.bmp'
            print("Testing face: " + facepath)
            found = find_face(facepath)
            if (person == found):
                success = success + 1

    print(success / 200 * 100)


def main():
    # store_models()
    # create_index()
    test()


if __name__ == "__main__":
    REDIS_URL = os.getenv("GABS_REDIS_URL")
    print("REDIS URL MAIN: {0}".format(REDIS_URL))
    main()
