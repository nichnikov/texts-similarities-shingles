import numpy as np
from itertools import islice

def shingle_split(splited_texts: [], shingle: int) -> []:
    """"""
    result = []
    for splited_text in splited_texts:
        if len(splited_text) <= shingle:
            result.append(["".join(splited_text)])
        else:
            shingles_list = [list(islice(splited_text, int(i), int(i + shingle))) for i in range(len(splited_text))]
            result.append(["".join(l) for l in shingles_list if len(l) == shingle])
    return result


def pairwise_sparse_jaccard_distance(X, Y=None):
    """
    Computes the Jaccard distance between two sparse matrices or between all pairs in
    one sparse matrix.

    Args:
        X (scipy.sparse.csr_matrix): A sparse matrix.
        Y (scipy.sparse.csr_matrix, optional): A sparse matrix.

    Returns:
        numpy.ndarray: A similarity matrix.
    """

    if Y is None:
        Y = X

    assert X.shape[1] == Y.shape[1]

    X = X.astype(bool).astype(int)
    Y = Y.astype(bool).astype(int)

    intersect = X.dot(Y.T)

    x_sum = X.sum(axis=1).A1
    y_sum = Y.sum(axis=1).A1
    xx, yy = np.meshgrid(x_sum, y_sum)
    union = ((xx + yy).T - intersect)

    return (1 - intersect / union).A
