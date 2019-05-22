import cPickle as pkl
import numpy as np

class DocLoader(object):
    def load_doc_vector(self, fname, dim_doc):
      #  fout = open("few.txt","w")
        vocab = {}
        with open(fname) as fin:
            f = np.load(fin)
            if dim_doc != len(f[0]):
                print("wrong doc dim")
            for line_no, docvector in enumerate(f):
                try:
                    vocab[line_no] = docvector
                   # print(docvector)
                   # fout.write(str(np.array(docvector)))
                  #  fout.write("\n")
                except:
                    print("Fewh:")
                    pass
        return vocab

