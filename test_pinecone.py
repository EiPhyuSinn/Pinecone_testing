import os
from pinecone import Pinecone,ServerlessSpec, CloudProvider, AwsRegion, Metric

api_key = os.environ.get("PINECONE_API_KEY")
if api_key is None:
    raise ValueError("PINECONE_API_KEY not found. Please set it in your environment.")

pc = Pinecone(api_key=api_key)

index_name = "eps-testing"

# if pc.has_index(index_name):
#     pc.delete_index(index_name)
#     print('deleted')

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        metric=Metric.COSINE,
        dimension=1024,
        spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),
    )

index = pc.Index(index_name)

# vector = pc.inference.embed(
#         model = "multilingual-e5-large",
#         inputs = ['THis is testing'],
#         parameters = {"input_type": "passage", "truncate": "END"}
# )

if pc.has_index(index_name):
    movies = [
        {
            "id": "0",
            "title": "Avatar",
            "summary": "On the alien world of Pandora, paraplegic Marine Jake Sully uses an avatar...",
            "year": 2009,
            "box_office": 2923706026
        },
        {
            "id": "1",
            "title": "Avengers: Endgame",
            "summary": "In the aftermath of Thanos wiping out half of the universe, the remaining Avengers...",
            "year": 2019,
            "box_office": 2799439100
        }
    ]

    for movie in movies:
        vector = pc.inference.embed(
            model = "multilingual-e5-large",
            inputs = [movie["summary"]],
            parameters = {"input_type": "passage", "truncate": "END"}
            )[0]

        index.upsert(vectors= [
            {
                "id": movie["id"],
                "values": vector.values,
                "metadata": {
                    "title": movie["title"],
                    "summary": movie["summary"],
                    "year": movie["year"],
                    "box_office": movie["box_office"]
                }
            }
        ],namespace= "happy-ending")


    movies = [
        {
            "id": "2",
            "title": "Ruth",
            "summary": "On the alien world of Pandora, paraplegic Marine Jake Sully uses an avatar...",
            "year": 2009,
            "box_office": 2923706026
        },
        {
            "id": "3",
            "title": "Final war",
            "summary": "In the aftermath of Thanos wiping out half of the universe, the remaining Avengers...",
            "year": 2019,
            "box_office": 2799439100
        }
    ]

    for movie in movies:
        vector = pc.inference.embed(
            model = "multilingual-e5-large",
            inputs = [movie["summary"]],
            parameters = {"input_type": "passage", "truncate": "END"}
            )[0]

        index.upsert(vectors= [
            {
                "id": movie["id"],
                "values": vector.values,
                "metadata": {
                    "title": movie["title"],
                    "summary": movie["summary"],
                    "year": movie["year"],
                    "box_office": movie["box_office"]
                }
            }
        ],namespace= "sad-ending")


query = "war"
query_embed = pc.inference.embed(
        model = "multilingual-e5-large",
        inputs = [query],
        parameters = {"input_type": "passage", "truncate": "END"}
        )[0].values

results = index.query(
    vector = query_embed,
    top_k = 5,
    include_metadata = True,
    namespace="sad-ending"
    )

print('Top matches')
for match in results['matches']:
    print(match['metadata']['title']," :", match['score'])
