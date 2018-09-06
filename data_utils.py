import os

import numpy as np

def load_embeddings(path, size, dimensions):

    embedding_matrix = np.zeros((size, dimensions), dtype=np.float32)

    # As embedding matrix could be quite big we 'stream' it into output file
    # chunk by chunk. One chunk shape could be [size // 10, dimensions].
    # So to load whole matrix we read the file until it's exhausted.
    size = os.stat(path).st_size
    with open(path, 'rb') as ifile:
        pos = 0
        idx = 0
        while pos < size:
            chunk = np.load(ifile)
            chunk_size = chunk.shape[0]
            embedding_matrix[idx:idx + chunk_size, :] = chunk
            idx += chunk_size
            pos = ifile.tell()
    return embedding_matrix[1:]
