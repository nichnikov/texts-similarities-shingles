import logging
from waitress import serve
from itertools import chain
from scipy.sparse import hstack
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, fields
from texts_processing import TextsTokenizer, QueriesVectors
from utils import pairwise_sparse_jaccard_distance, shingle_split


logger = logging.getLogger("app_shingles")
logger.setLevel(logging.INFO)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)

name_space = api.namespace('api', 'На вход поступает JSON, возвращает JSON')

input_data = name_space.model("Input JSONs",
                              {"Shingle_len": fields.Integer(required=True, help="Длина шингла"),
                               "Texts1": fields.List(fields.String(required=True, help="Тексты для сравнения 1")),
                               "Texts2": fields.List(fields.String(required=True, help="Тексты для сравнения 1")),
                               "Score": fields.Float(required=True, help="Степень похожести")})

tokenizer = TextsTokenizer()


@name_space.route('/')
class Shingles(Resource):
    """"""

    @name_space.expect(input_data)
    def post(self):
        json_data = request.json
        tx1 = json_data["Texts1"]
        tx2 = json_data["Texts2"]
        shingles_len = json_data["Shingle_len"]
        score = json_data["Score"]

        if tx1 and tx2:
            lm_tx1 = tokenizer([str(x) for x in tx1])
            lm_tx2 = tokenizer([str(x) for x in tx2])

            t1_shingles = shingle_split(lm_tx1, shingles_len)
            t2_shingles = shingle_split(lm_tx2, shingles_len)
            tokens_quantity = len(set([x for x in chain(*t1_shingles)] + [y for y in chain(*t2_shingles)]))

            vectorizer = QueriesVectors(tokens_quantity)
            t1_vectors = vectorizer(t1_shingles)
            t2_vectors = vectorizer(t2_shingles)

            matrix1 = hstack(t1_vectors).T
            matrix2 = hstack(t2_vectors).T

            # jaccard_distance = pairwise_sparse_jaccard_distance(matrix1, matrix2)
            jaccard_matrix = 1 - pairwise_sparse_jaccard_distance(matrix1, matrix2)

            print("tx1:", tx1[:10])
            print("\n\ntx2:", tx2[:10])
            print("\n\nlm_tx1:", lm_tx1[:10])
            print("\n\nlm_tx2:", lm_tx2[:10])
            print(len(tx1))
            print(len(lm_tx1))
            print(len(lm_tx2))
            print(jaccard_matrix.shape)
            indexes = (jaccard_matrix > score).nonzero()
            search_results = [(tx1[i], tx2[j], jaccard_matrix[i][j]) for i, j in zip(indexes[0], indexes[1])]
        else:
            logger.warning("No input texts")
            search_results = []

        return jsonify({"shingle_duplicates": search_results})


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=6011)
    # app.run(host='0.0.0.0', port=6011)
