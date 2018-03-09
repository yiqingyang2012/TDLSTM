# -*- coding: utf-8 -*
from glove import Glove
from gensim import interfaces, utils
from nltk.corpus import stopwords
import nltk
import tensorflow as tf
import numpy as np
import chardet
import unicodedata

english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']

stopwords_eng = stopwords.words('english')


def convertunicodeToAssic():
	global stopwords_eng
	tmpstoplist = []
	for word in stopwords_eng:
		tmpstoplist.append(unicodedata.normalize('NFKD',word).encode('ascii','ignore'))
	stopwords_eng = tmpstoplist+english_punctuations

def test_stanford_loading(file):
	model = Glove.load_stanford(file)
	return (model.word_vectors, model.dictionary)

	
def loadDataSet(word_to_id, file, FLAGS):
	global stopwords_eng
	dataset = []
	num = 0
	printid =0
	id_to_word = {v: k for k, v in word_to_id.items()}
	print ("load data")

	fin = open(file,'r')
	lines = fin.readlines()
	lineSize = len(lines)

	if lineSize % 4 != 0:
		print ("error")
		exit()

	leftLen =0
	rightLen = 0
	aspectLen = 0
	printid = 0

	for inx in range(0, lineSize//4):
		line4 = lines[4*inx:4*(inx+1)]

		if line4[0].strip() == '':
			left = ['<BOS>']
		else:
			left = line4[0].strip().split(' ')
			left = ['<BOS>'] + [w for w in left if w not in stopwords_eng]
		leftLen = len(left)
		left = wordToId(left, word_to_id, 0, FLAGS)

		if line4[0].strip() == '':
			right = ['<EOS>']
		else:
			right = line4[0].strip().split(' ')
			right = [w for w in right if w not in stopwords_eng] + ['<EOS>']
		leftLen = len(right)
		right = wordToId(right, word_to_id, 0, FLAGS)

		filtered = line4[2].strip().split(' ')
		filtered = [w for w in filtered if w not in stopwords_eng]
		aspects = wordToId(filtered, word_to_id, 1, FLAGS)
		aspectLen = len(aspects)

		polarity = int(line4[3].strip())

		dataSample = DataInstance(left, leftLen, right, rightLen, aspects, aspectLen, polarity) 
		dataset.append(dataSample)

	print ("finished")
	return dataset

def loadDataSet_1(word_to_id, file, FLAGS):
	global stopwords_eng
	dataset = []
	num = 0
	printid =0
	id_to_word = {v: k for k, v in word_to_id.items()}

	with utils.smart_open(file, 'r') as fin:
		right =[]
		left = []
		aspects = []
		polarity = 0
		leftLen =0
		rightLen = 0
		aspectLen = 0
		i=0
		bad_ind = 0
		for lineno, line in enumerate(fin):
			words = line.strip()
			words = words.split(' ')
			i=i+1
			if num == 0:
				filtered = [w for w in words if w not in stopwords_eng]
				leftLen = len(filtered)
				if leftLen==0:
					bad_ind = 1
				#assert leftLen != 0, print("leftLen == 0 left: {0} ***filter: {1}.  len={2} , i={3}".format(words, filtered, leftLen, i))
				left = wordToId(filtered, word_to_id, 0, FLAGS)
				num = num+1
			elif num == 1:
				filtered = [w for w in words if w not in stopwords_eng]
				rightLen = len(filtered)
				if rightLen==0:
					bad_ind =1
				#assert rightLen != 0, print("rightLen == 0 right : {0} ,***filter{1}".format(words, filtered))
				revert_words = [0]*len(filtered)
				j = 1
				for word in filtered:
					revert_words[-1 *j] = word
					j +=1
				right = wordToId(revert_words, word_to_id, 0, FLAGS)
				num = 2
			elif num == 2:
				filtered = [w for w in words if w not in stopwords_eng]
				aspectLen = len(filtered)
				if aspectLen == 0:
					bad_ind = 1
				#assert aspectLen != 0, print("aspect len == 0, aspect: {0},*** filter: {1}".format(words,filtered))
				aspects = wordToId(filtered, word_to_id, 1, FLAGS)
				num = 3
			else:
				if bad_ind == 0:
					polarity = int(words[0])
					dataSample = DataInstance(left, leftLen, right, rightLen, aspects, aspectLen, polarity) 
					dataset.append(dataSample)
					if printid== 0:
						print("frist time")
						print([id_to_word[aspect] for aspect in dataSample.aspects])
						print(dataSample.polarity)
						printid =1
				num = 0
				right =[]
				left = []
				aspects = []
				polarity = 0
				leftLen =0
				rightLen = 0
				aspectLen = 0
	return dataset

def statistic_vocabulary(FLAGS):
	global stopwords_eng
	num = 0
	wordset = set()
	fin = open(FLAGS.train_file,'r')
	lines = fin.readlines()
	lineSize = len(lines)

	for inx in range(0, lineSize//4):
		line4 = lines[4*inx:4*(inx+1)]
		tmpline = []
		for line in line4:
			tmpline.extend(line.strip().split(' '))

		for word in tmpline:
			#print chardet.detect(word)
			if word not in stopwords_eng:
				wordset.add(word)
	fin.close()

	fin = open(FLAGS.test_file,'r')
	for inx in range(0, lineSize//4):
		line4 = lines[4*inx:4*(inx+1)]
		tmpline = []
		for line in line4:
			tmpline.extend(line.strip().split(' '))

		for word in tmpline:
			#print chardet.detect(word)
			if word not in stopwords_eng:
				wordset.add(word)
	fin.close()

	return wordset





def wordToId(words, wordtoid, type, FLAGS):
	tmp = []
	#filter stop words
	for word in words:
		if word in wordtoid:
			tmp.append(wordtoid[word])
		else:
			tmp.append(wordtoid['<UNK>'])
	if type == 0:
		tmp += [wordtoid['<PAD>']]*(FLAGS.max_article_len-len(words))
	else:
		tmp += [wordtoid['<PAD>']]*(FLAGS.max_target_len-len(words))

	return tmp	


class DataInstance(object):
 	"""docstring for DataInstance"""
 	def __init__(self, left, leftlen, right, rightlen, aspects, aspectlen, polarity):
 		self.left = left
 		self.leftLen = leftlen
 		self.right = right
 		self.rightLen = rightlen
 		self.aspects = aspects
 		self.aspectlen = aspectlen
 		self.polarity =polarity

def load_word2vec(w2v_file, id_to_word):
    with open(path, "rb") as f:
        emb_and_dict = test_stanford_loading(w2v_file)
        word_vec = emb_and_dict[0]
        dict = emb_and_dict[1]
        word2vec = []
        for i, word in id_to_word.items():
            if word in word_vec:
                word2vec.append(word_vec[word])
            else:
                word2vec.append(word_vec["<UNK>"])
    return word2vec

'''
# path for log, model and result
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("embeding_file", './log', "path for embeding files")
tf.app.flags.DEFINE_string("train_file", './weights', "path to train model")
tf.app.flags.DEFINE_string("test_file", './weights', "path to test file")
tf.app.flags.DEFINE_integer("max_article_len", 150, "maximum sentencelen")
tf.app.flags.DEFINE_integer("embedingDim", 100, "word dimention")
tf.app.flags.DEFINE_integer("max_target_len", 5, "max aspect len")

def main(_):
 	embeding_dict = test_stanford_loading(FLAGS.embeding_file)
 	print type(embeding_dict[0])
 	convertunicodeToAssic()
 	experiment_vocabulary = statistic_vocabulary(FLAGS)

 	voc = dict()
 	indx = 0

 	embeding_file = np.ndarray([len(experiment_vocabulary),FLAGS.embedingDim], dtype='float64')
 	print("embeding shape:")
 	print(len(experiment_vocabulary))
 	print(FLAGS.embedingDim)
 	print(len(embeding_dict[0][embeding_dict[1]['unknow']]))

 	for word in experiment_vocabulary:
 		if word in embeding_dict[1]:
 			voc[word] = indx
 			embeding_file[indx] = embeding_dict[0][embeding_dict[1][word]]
 			indx += 1
 	voc['unknow'] = indx
 	embeding_file[indx] = embeding_dict[0][embeding_dict[1]['unknow']]
 	indx +=1
 	voc['<PAD>'] = indx
 	embeding_file[indx] = np.zeros(FLAGS.embedingDim, dtype='float64')

 	train_dataset = loadDataSet_1(voc, FLAGS.train_file, FLAGS)
 	test_dataset = loadDataSet_1(voc, FLAGS.test_file, FLAGS)

if __name__ == "__main__":
	tf.app.run(main)
'''
def initia_data(FLAGS):
 	embeding_dict = test_stanford_loading(FLAGS.embeding_file)
 	experiment_vocabulary = statistic_vocabulary(FLAGS)

 	voc = dict()
 	indx = 0

 	embeding_file = np.ndarray([len(experiment_vocabulary),FLAGS.embedingDim], dtype='float64')
 	print("embeding shape:")
 	print(len(experiment_vocabulary))
 	print(FLAGS.embedingDim)
 	print(len(embeding_dict[0][embeding_dict[1]['unknow']]))

 	for word in experiment_vocabulary:
 		if word in embeding_dict[1]:
 			voc[word] = indx
 			embeding_file[indx] = embeding_dict[0][embeding_dict[1][word]]
 			indx += 1
 	voc['<UNK>'] = indx
 	embeding_file[indx] = np.random.random((FLAGS.embedingDim,))
 	indx +=1
 	voc['<PAD>'] = indx
 	embeding_file[indx] = np.random.random((FLAGS.embedingDim,))
 	indx +=1
 	voc['<BOS>'] = indx
 	embeding_file[indx] = np.random.random((FLAGS.embedingDim,))
 	indx +=1
 	voc['<EOS>'] = indx
 	embeding_file[indx] = np.random.random((FLAGS.embedingDim,))
 	train_dataset = loadDataSet(voc, FLAGS.train_file, FLAGS)
 	test_dataset = loadDataSet(voc, FLAGS.test_file, FLAGS)
 	return train_dataset, test_dataset, embeding_file, voc
 	'''
 	embeding_dict = test_stanford_loading(FLAGS.embeding_file)
 	convertunicodeToAssic()
 	train_dataset = loadDataSet(embeding_dict[1], FLAGS.train_file, FLAGS)
 	test_dataset = loadDataSet(embeding_dict[1], FLAGS.test_file, FLAGS)
 	return train_dataset, test_dataset, embeding_dict[0], embeding_dict[1]
 	'''
