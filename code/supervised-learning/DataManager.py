import numpy as np
import theano
import cPickle as pkl

class Sentence(object):
    """docstring for sentence"""
    def __init__(self, content, target, rating, grained):
        self.content, self.target = content.lower(), target
        self.solution = np.zeros(grained, dtype=theano.config.floatX)
        self.senlength = len(self.content.split(' '))
        try:
            self.solution[int(rating)] = 1#
        except:
            exit()
    def stat(self, wordlist, grained=2):
        data, data_target, i = [], [], 0
        solution = np.zeros((self.senlength, grained), dtype=theano.config.floatX)
        for word in self.content.split(' '):
            if word in wordlist:
                data.append(wordlist[word])
            else:
                data.append(0)
            try:
                pol = Lexicons_dict[word]
                solution[i][pol+1] = 1
            except:
                pass
            i = i+1
        for word in self.target.split(' '):
            data_target.append(word)
        return {'seqs': data, 'target': data_target, 'solution': np.array([self.solution]), 'target_index': int(self.target)}

class DataManager(object):
    def __init__(self, dataset, grained=2):
        self.fileList = ['train', 'test']
        self.origin = {}
        for fname in self.fileList:
            data = []
            with open('%s/imdb_%s.txt' % (dataset, fname)) as f:
                sentences = f.readlines()
                for i in range(len(sentences)/3):
                    target, content, rating = sentences[i*3].strip(), sentences[i*3+1].strip(), sentences[i*3+2].strip()
                    sentence = Sentence(content, target, rating, grained)                    
                    if fname == 'train' and len(content.split(' ')) < 800:
                        data.append(sentence)
                    if fname == 'test':
                        data.append(sentence)
            self.origin[fname] = data

    def gen_word(self):
        wordcount = {}
        def sta(sentence):
            for word in sentence.content.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except:
                    wordcount[word] = 1
            for word in sentence.target.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except:
                    wordcount[word] = 1

        fname = self.fileList[0]
        for sent in self.origin[fname]:
            sta(sent)
        words = wordcount.items()
        words.sort(key=lambda x:x[1], reverse=True)
        if len(words) > 30000:
            words = words[:30000]
        self.wordlist = {item[0]:index+1 for index, item in enumerate(words)}
        return self.wordlist
    def load_word(self,path):
        self.wordlist = pkl.load(open(path,"rb"))
        return self.wordlist        
    def gen_data(self, grained=2):
        self.data = {}
        for fname in self.fileList:
            self.data[fname] = []
            for sent in self.origin[fname]:
                self.data[fname].append(sent.stat(self.wordlist))
        return self.data['train'], self.data['test']

    def word2vec_pre_select(self, mdict, word2vec_file_path, save_vec_file_path):
        list_seledted = ['']
        line = ''
        with open(word2vec_file_path) as f:
            for line in f:
                tmp = line.strip().split(' ', 1)
                if mdict.has_key(tmp[0]):
                    list_seledted.append(line.strip())
        list_seledted[0] = str(len(list_seledted)-1) + ' ' + str(len(line.strip().split())-1)
        open(save_vec_file_path, 'w').write('\n'.join(list_seledted))
