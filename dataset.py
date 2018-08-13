from PreProcessing import PreProcessing
import numpy as np
import xml.etree.ElementTree as ET
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import models
from torch.utils.data import dataset, dataloader


def point_wise_add(x, y):
    result = []
    assert len(x) == len(y)
    for idx in range(len(x)):
        result.append(x[idx] + y[idx])
    return result


def point_wise_compare(x, y):
    # x <= y
    assert len(x) == len(y)
    for idx in range(len(y)):
        if x[idx] > y[idx]:
            return False
    return True


def scaler_vector_mult(s, v):
    result = []
    for idx in range(len(v)):
        result.append(int(s * v[idx]))
    return result


class SimpleDataset():
    def __init__(self, train_file, test_file, validation_percentage):
        self.category_label_num = {
            'service': 0,
            'food': 1,
            'price': 2,
            'ambience': 3,
            'anecdotes/miscellaneous': 4
        }
        self.validation_percentage = validation_percentage
        self.extract_data(train_file, test_file)

    def __len__(self):
        return len(self.train_data)

    def get_sentences(self, cat):
        result = []
        for data_instance in self.train_data:
            # if data_instance[1][self.category_label_num[cat]] != 1:
            #     continue
            sentence = ' '.join(data_instance[0])
            for word in sentence:
                if word.isdigit():
                    sentence = sentence.replace(word, ' ')
            sentence = sentence.split()
            result.append(sentence)
        return result

    def get_sentences_by_partial_label(self, p_label):
        result = []
        for data_instance in self.train_data:
            if p_label.lower() not in data_instance[1].lower():
                continue
            sentence = ' '.join(data_instance[0])
            for word in sentence:
                if word.isdigit():
                    sentence = sentence.replace(word, ' ')
            sentence = sentence.split()
            if sentence not in result:
                result.append(sentence)
        return result

    def get_labels(self, cat):
        result = []
        for data_instance in self.train_data:
            if data_instance[1][self.category_label_num[cat]] == 1:
                result.append(1)
            else:
                result.append(0)
        return result

    def extract_data(self, train_file, test_file):
        tree = ET.parse(train_file)
        root = tree.getroot()
        train_sentences = root.findall('sentence')
        tree = ET.parse(test_file)
        root = tree.getroot()
        test_sentences = root.findall('sentence')
        self.train_sentence_with_all_labels = {}
        self.train_labels = {}
        self.test_labels = {}
        self.processed_train_sentences = self.process_data(train_sentences)
        self.processed_test_sentences = self.process_data(test_sentences)
        self.original_train_sentences = self.getOriginalsentences(train_sentences)
        self.original_test_sentences = self.getOriginalTestsentences(test_sentences)
        self.train_data, self.categories, self.valid_data = self.get_inputs(self.processed_train_sentences,
                                                           train_sentences,
                                                           is_train=True)
        self.test_data = self.get_inputs(self.processed_test_sentences,
                                         test_sentences)
        self.number_of_categories = len(self.categories)
        print(self.categories)
        print(len(self.train_data))

    def getOriginalsentences(self, unprocessed_data):
        unprocessed_sentences = []
        for sentence in unprocessed_data:
            text = sentence[0].text
            if '$' in text:
                text = text.replace('$', ' price ')
            text = text.lower()
            unprocessed_sentences.append(text)
        return unprocessed_sentences

    def getOriginalTestsentences(self, unprocessed_data):
        unprocessed_sentences = []
        for sentence in unprocessed_data:
            text = sentence[0].text
            if '$' in text:
                text = text.replace('$', ' price ')
            text = text.lower()
            unprocessed_sentences.append(text)
        return unprocessed_sentences

    def process_data(self, unprocessed_data):
        unprocessed_sentences = []
        for sentence in unprocessed_data:
            text = sentence[0].text
            if '$' in text:
                text = text.replace('$', ' price ')
            text = text.lower()
            unprocessed_sentences.append(text)
        preprocessor = PreProcessing(unprocessed_sentences, 'english')
        preprocessor.Remove_Punctuation()
        processed_sentences = preprocessor.Remove_StopWords()

        return processed_sentences

    def get_inputs(self, processed_sentences, unprocessed_data, is_train=False):
        num_of_train_sentences = 3000
        processed_data = []
        categories = []
        valid_data = []
        valid_size = 0
        train_data = []
        for i in range(len(processed_sentences)):
            processed_sentences[i] = processed_sentences[i].split()
        for i in range(len(unprocessed_data)):
            sentence = processed_sentences[i]
            sentence_categories = []
            if len(unprocessed_data[i]) > 1 and len(unprocessed_data[i][1]) > 0:
                if unprocessed_data[i][1].tag == 'aspectCategories':
                    aspect_cats = unprocessed_data[i][1]
                else:
                    aspect_cats = unprocessed_data[i][2]
                if valid_size < self.validation_percentage * num_of_train_sentences:
                    add_valid_data = True
                    valid_size += 1
                else:
                    add_valid_data = False
                if is_train:
                    labels = 5 * [0]
                    for opinions in aspect_cats:
                        dict = opinions.attrib
                        if dict['category'] in sentence_categories:
                            continue
                        labels[self.category_label_num[str(dict['category'])]] = 1
                        sentence_categories.append(dict['category'])
                        if dict['category'] not in categories:
                            categories.append(dict['category'])
                    if add_valid_data:
                        if [sentence, labels] not in valid_data:
                            valid_data.append([sentence, labels])
                    else:
                        if [sentence, labels] not in train_data:
                            train_data.append([sentence, labels])
                else:
                    test_sentence_categories = []
                    # sentence_categories = []
                    if unprocessed_data[i][1].tag == 'aspectCategories':
                        aspect_cats = unprocessed_data[i][1]
                    else:
                        aspect_cats = unprocessed_data[i][2]
                    for opinions in aspect_cats:
                        dict = opinions.attrib
                        if self.category_label_num[dict['category']] not in test_sentence_categories:
                            test_sentence_categories.append(self.category_label_num[dict['category']])
                            sentence_categories.append(dict['category'])
                    processed_data.append([sentence, test_sentence_categories])
                    self.test_labels[i] = sentence_categories

            else:
                if is_train:
                    processed_data.append([sentence, 'NULL'])
                    self.train_sentence_with_all_labels[' '.join(sentence)] = 5 * [0]
                    self.train_sentence_with_all_labels[' '.join(sentence)][-1] = 1
                else:
                    processed_data.append([sentence, [self.category_label_num['NULL']]])
        if is_train:
            return train_data, categories, valid_data
        else:
            return processed_data


word_em_sg = models.KeyedVectors.load_word2vec_format('../yelp_W2V_skipgram.bin', binary=True)
word_em_glov = models.KeyedVectors.load_word2vec_format('../yelp_glove.bin', binary=True)
word_em_cbow = models.KeyedVectors.load_word2vec_format('../yelp_W2V_300_orig.bin', binary=True)
# un_words_ft = {}
# un_words_glov = {}
# un_words_w2v = {}

simple_dataset = None


class DataLoader(dataset.Dataset):
    def __init__(self, data='train', word_embeddings=['cbow', 'sg', 'glove'], simple_dataset=None):
        assert simple_dataset != None
        self.simple_dataset = simple_dataset
        if data == 'train':
            self.data = self.simple_dataset.train_data
        elif data == 'test':
            self.data = self.simple_dataset.test_data
        else:
            self.data = self.simple_dataset.valid_data
        self.word_embeddings = word_embeddings
        self.sentence_length = 35
        self.data_type = data

    def __len__(self):
        return len(self.data)

    def get_num_of_classes(self):
        return self.simple_dataset.number_of_categories

    def __getitem__(self, item):
        sentence = []
        for word in self.data[item][0]:
            word_rep = []
            try:
                sg_word_rep = word_em_sg[word]
            except KeyError:
                continue
            try:
                glov_word_rep = word_em_glov[word]
            except KeyError:
                continue
            try:
                cbow_word_rep = word_em_cbow[word]
            except KeyError:
                continue
            if 'sg' in self.word_embeddings:
                word_rep += list(sg_word_rep)
            if 'glove' in self.word_embeddings:
                word_rep += list(glov_word_rep)
            if 'cbow' in self.word_embeddings:
                word_rep += list(cbow_word_rep)
            sentence.append(np.array(word_rep))
        while len(sentence) < self.sentence_length:
            if len(self.word_embeddings) == 1:
                sentence.append(np.array(np.zeros(300), dtype='float64'))
            elif len(self.word_embeddings) == 2:
                sentence.append(np.array(np.zeros(600), dtype='float64'))
            elif len(self.word_embeddings) == 3:
                sentence.append(np.array(np.zeros(900), dtype='float64'))
        sentence = np.array(sentence, dtype='float64')
        label = np.array(self.data[item][1])
        # entity_label = len(self.simple_dataset.entity) * [0]
        # attrib_label = len(self.simple_dataset.attrib) * [0]
        # entity_label = np.array(entity_label)
        # attrib_label = np.array(attrib_label)
        return sentence, np.array([label])#, np.array([]), np.array([])


if __name__ == '__main__':
    dataset = dataset = SimpleDataset('../datas/ABSA14_Restaurants_Train.xml',
                                      '../datas/Restaurants_Test_Data_phaseB.xml', validation_percentage=0.1)
    train_loader = DataLoader(data='train', simple_dataset=dataset)
    valid_loader = DataLoader(data='valid', simple_dataset=dataset)
    test_loader = DataLoader(data='test', simple_dataset=dataset)
    print(train_loader[0])