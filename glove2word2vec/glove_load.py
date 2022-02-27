from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim

'''Use gensim change glove to word2vector form'''
glove_file = 'glove.6B.50d.txt'  # Need to download the glove.6B.50d.txt file first
tmp_file = 'glove_word2vec.txt'

glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)

model.save_word2vec_format(tmp_file, binary=False)

# --- test model ---
# model = gensim.models.KeyedVectors.load_word2vec_format(tmp_file, binary=False)
# print(model['the'])
