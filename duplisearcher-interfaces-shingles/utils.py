import io
import requests
import pandas as pd
from flask import Response
from werkzeug.datastructures import Headers
from werkzeug.datastructures import FileStorage

"""
def api_configurator(name_space):
    """"""
    upload_parser = name_space.parser()
    upload_parser.add_argument("file", type=FileStorage, location='files', required=True)
    upload_parser.add_argument("score", type=float, required=True)
    return upload_parser"""


def remote_duplisearcher(args, sending_url, upload_type="excel"):
    """"""
    if upload_type == "json":
        json_data = args
    else:
        if upload_type == "excel":
            df1 = pd.read_excel(args['file1'], header=None)
            df2 = pd.read_excel(args['file2'], header=None)

        else:
            """The function expects csv type of upload data"""
            df1 = pd.read_csv(args['file1'], header=None)
            df2 = pd.read_csv(args['file2'], header=None)

        json_data = {"Texts1": list(set(df1[0])), "Texts2": list(set(df2[0])), "Score": args['score'],
                     "Shingle_len": args["shingle_len"]}

    response = requests.post(sending_url, json=json_data)
    results = response.json()
    return pd.DataFrame(results["shingle_duplicates"], columns=["text1", "text2", "jaccard_coeff"])


def response_func(clustering_texts_df, response_type="excel"):
    """"""
    headers = Headers()
    if response_type == "excel":
        headers.add('Content-Disposition', 'attachment', filename="clustering_results.xlsx")
        buffer = io.BytesIO()
        clustering_texts_df.to_excel(buffer, index=False, encoding='cp1251'),
        return Response(buffer.getvalue(),
                        mimetype='application/vnd.ms-excel',
                        headers=headers)

    else:
        headers.add('Content-Disposition', 'attachment', filename="clustering_results.csv")
        return Response(clustering_texts_df.to_csv(index=False),
                        mimetype="text/csv",
                        headers=headers)
