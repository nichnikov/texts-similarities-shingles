import os
from flask_cors import CORS
from dotenv import load_dotenv
from flask import Flask, request
from werkzeug.datastructures import FileStorage
from flask_restplus import Api, Resource, fields
from utils import remote_duplisearcher, response_func
from waitress import serve

load_dotenv(".env")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)

CORS(app, resources={r"/api/*": {"enabled": "true",
                                 "headers": [
                                     "Authorization",
                                     "Origin",
                                     "Access-Control-Request-Method",
                                     "Access-Control-Request-Headers",
                                     "Accept",
                                     "Accept-Charset",
                                     "Accept-Encoding",
                                     "Accept-Language",
                                     "Cache-Control",
                                     "Content-Type",
                                     "Cookie",
                                     "DNT",
                                     "Pragma",
                                     "Referer",
                                     "User-Agent",
                                     "X-Forwarded-For"
                                 ]
                                 }}, supports_credentials=True)

name_space = api.namespace('api', 'three interfaces *.json, *.csv, *.xlsx')

input_data = name_space.model("Insert JSON",
                              {"texts": fields.List(fields.String(description="Insert texts")),
                               "scores_list": fields.List(fields.Float(description="Distance", required=True))}, )

DUPLISEARCHER_URL = os.environ.get("DUPLISEARCHER_URL")
if DUPLISEARCHER_URL is None: raise Exception('Env var DUPLISEARCHER_URL not defined')


# CLUSTERING_URL = "http://0.0.0.0:7001/api/"

@name_space.route('/json_excel')
class DuplisearcherJsonExcel(Resource):
    @name_space.expect(input_data)
    def post(self):
        """POST method on input json file with texts and score, output clustering texts as xlsx file."""
        json_data = request.json
        clustering_texts_df = remote_duplisearcher(json_data, DUPLISEARCHER_URL, upload_type="json")
        return response_func(clustering_texts_df)


"""================== csv files ==================="""


def api_configurator(name_space):
    """"""
    upload_parser = name_space.parser()
    upload_parser.add_argument("file1", type=FileStorage, location='files', required=True,
                               help="???????????? ???????????? ???????? ?? ???????????? ??????????????")
    upload_parser.add_argument("file2", type=FileStorage, location='files', required=True,
                               help="???????????? ???????????? ???????? ?? ???????????? ??????????????")
    upload_parser.add_argument("shingle_len", type=float, required=True,
                               help="?????????? ??????????????")
    upload_parser.add_argument("score", type=float, required=True,
                               help="???????????????? ??????????")
    return upload_parser


# https://debugthis.dev/flask/2019-10-09-building-a-restful-api-using-flask-restplus/
csv_name_space = api.namespace('api', '???????????????? ?????????? csv ?? ???????????????? (?????????? ?????? ?????????????????????????? ?? ???????????? ??????????????)'
                                      ' ?? score (??????????)')
csv_upload_parser = api_configurator(csv_name_space)


@csv_name_space.route('/csv_csv')
@csv_name_space.expect(csv_upload_parser)
class DuplisearcherCsvCsv(Resource):
    def post(self):
        """POST method on input csv file with texts and score, output clustering texts  as csv file."""
        args = csv_upload_parser.parse_args()
        clustering_texts_df = remote_duplisearcher(args, DUPLISEARCHER_URL, upload_type="csv")
        return response_func(clustering_texts_df, response_type="csv")


"""================== excel files ==================="""
# https://debugthis.dev/flask/2019-10-09-building-a-restful-api-using-flask-restplus/
excel_name_space = api.namespace('api', '???????????????? ?????????? excel ?? ???????????????? (?????????? ?????? ?????????????????????????? ?? ???????????? ??????????????)'
                                        ' ?? score (??????????)')
excel_upload_parser = api_configurator(excel_name_space)


@excel_name_space.route('/excel_excel')
@excel_name_space.expect(excel_upload_parser)
class DuplisearcherEcxelExcel(Resource):
    def post(self):
        """POST method on input xlsx file with texts and score, output clustering texts  as xlsx file."""
        args = excel_upload_parser.parse_args()
        clustering_texts_df = remote_duplisearcher(args, DUPLISEARCHER_URL)
        return response_func(clustering_texts_df)


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=6010)
    # app.run(host='0.0.0.0', port=6010)
