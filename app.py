from flask_api import FlaskAPI
from flask import request, url_for
import sentencepiece as spm
from gensim.models import KeyedVectors
import numpy as np

#fasttext_location = '/home/vadim/data/wiki.simple.bin'
bpe_model_location = 'ru.wiki.bpe.op200000.model'
bpe_vec_location = 'ru.wiki.bpe.op200000.d300.w2v.bin'

sp = spm.SentencePieceProcessor()
sp.Load(bpe_model_location)
model = KeyedVectors.load_word2vec_format(bpe_vec_location, binary=True)
app = FlaskAPI(__name__)

@app.route('/')
def hello():
    return {'hello': 'world'}

@app.route("/embed", methods=['GET', 'POST'])
def embed():
    if request.method == 'POST':
        text = request.data.get('text')
        pieces = sp.encode_as_pieces(text)
        embedding = np.zeros(model.wv.vector_size)
        piece_count = 0

        for binary_piece in pieces:
            piece = binary_piece.decode('utf-8')
            try:
                embedding += model[piece]
                piece_count += 1
            except KeyError:
                pass

        if piece_count:
            embedding /= piece_count

        return {'embedding': embedding.tolist()}
    else:
        return {'Instruction': 'Make a POST request with json body and "text" attribute'}