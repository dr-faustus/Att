from PreProcessing import PreProcessing
import numpy as np
import xml.etree.ElementTree as ET
from gensim import models
from torch.utils.data import dataset, dataloader


class SimpleDataset:
    def __init__(self, train_file, test_file, validation_percentage):
        self.category_label_num = {
            'RESTAURANT#GENERAL': 0,
            'SERVICE#GENERAL': 1,
            'FOOD#QUALITY': 2,
            'FOOD#STYLE_OPTIONS': 3,
            'DRINKS#STYLE_OPTIONS': 4,
            'DRINKS#PRICES': 5,
            'RESTAURANT#PRICES': 6,
            'RESTAURANT#MISCELLANEOUS': 7,
            'AMBIENCE#GENERAL': 8,
            'FOOD#PRICES': 9,
            'LOCATION#GENERAL': 10,
            'DRINKS#QUALITY': 11,
        }
        self.category_num_label = {
            0: 'RESTAURANT#GENERAL',
            1: 'SERVICE#GENERAL',
            2: 'FOOD#QUALITY',
            3: 'FOOD#STYLE_OPTIONS',
            4: 'DRINKS#STYLE_OPTIONS',
            5: 'DRINKS#PRICES',
            6: 'RESTAURANT#PRICES',
            7: 'RESTAURANT#MISCELLANEOUS',
            8: 'AMBIENCE#GENERAL',
            9: 'FOOD#PRICES',
            10: 'LOCATION#GENERAL',
            11: 'DRINKS#QUALITY',
        }
        self.entity = [
            'RESTAURANT',
            'SERVICE',
            'FOOD',
            'DRINKS',
            'AMBIENCE',
            'LOCATION',
        ]
        self.attrib = [
            'GENERAL',
            'QUALITY',
            'STYLE_OPTIONS',
            'MISCELLANEOUS',
            'PRICES',
        ]
        self.entity_label_num = {
            'RESTAURANT': 0,
            'FOOD': 1,
            'DRINKS': 2,
            'AMBIENCE': 3,
            'SERVICE': 4,
            'LOCATION': 5,
        }
        self.attrib_label_num = {
            'GENERAL': 0,
            'QUALITY': 1,
            'STYLE_OPTIONS': 2,
            'PRICES': 3,
            'MISCELLANEOUS': 4,
        }
        self.category_num_entity = {
            0: 'RESTAURANT',
            1: 'SERVICE',
            2: 'FOOD',
            3: 'FOOD',
            4: 'DRINKS',
            5: 'DRINKS',
            6: 'RESTAURANT',
            7: 'RESTAURANT',
            8: 'AMBIENCE',
            9: 'FOOD',
            10: 'LOCATION',
            11: 'DRINKS',
        }
        self.category_num_attrib = {
            0: 'GENERAL',
            1: 'GENERAL',
            2: 'QUALITY',
            3: 'STYLE_OPTIONS',
            4: 'STYLE_OPTIONS',
            5: 'PRICES',
            6: 'PRICES',
            7: 'MISCELLANEOUS',
            8: 'GENERAL',
            9: 'PRICES',
            10: 'GENERAL',
            11: 'QUALITY',
        }
        self.validation_percentage = validation_percentage
        self.extract_data(train_file, test_file)

    def get_sentences(self, cat):
        result = []
        for data_instance in self.train_data:
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
        train_sentences = root.findall('**/sentence')
        tree = ET.parse(test_file)
        root = tree.getroot()
        test_sentences = root.findall('**/sentence')
        self.train_sentence_with_all_labels = {}
        self.processed_train_sentences = self.process_data(train_sentences)
        self.processed_test_sentences = self.process_data(test_sentences)
        self.train_data, self.categories, self.valid_data = self.get_inputs(self.processed_train_sentences,
                                                                            train_sentences,
                                                                            is_train=True)
        self.test_data = self.get_inputs(self.processed_test_sentences,
                                         test_sentences)
        print(self.categories)
        self.number_of_categories = len(self.categories)

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
        num_of_train_sentences = 1708
        processed_data = []
        categories = []
        for i in range(len(processed_sentences)):
            processed_sentences[i] = processed_sentences[i].split()
        num_of_data_per_label = [0] * len(self.category_label_num.keys())
        valid_data = []
        valid_size = 0
        train_data = []
        for i in range(len(unprocessed_data)):
            sentence = processed_sentences[i]
            sentence_categories = []
            sentence_attrib = unprocessed_data[i].attrib
            try:
                if sentence_attrib['OutOfScope'] == 'TRUE':
                    continue
            except KeyError:
                pass
            if len(unprocessed_data[i]) > 1:
                if is_train:
                    if len(unprocessed_data[i][1]) == 0:
                        continue
                    if valid_size < self.validation_percentage * num_of_train_sentences:
                        add_valid_data = True
                        valid_size += 1
                    else:
                        add_valid_data = False
                    labels = len(self.category_label_num.keys()) * [0]
                    for opinions in unprocessed_data[i][1]:
                        dict = opinions.attrib

                        if str(dict['category']) not in categories:
                            categories.append(str(dict['category']))

                        labels[self.category_label_num[str(dict['category'])]] = 1
                        sentence_categories.append(dict['category'])
                        num_of_data_per_label[self.category_label_num[str(dict['category'])]] += 1
                    if add_valid_data:
                        if [sentence, labels] not in valid_data:
                            valid_data.append([sentence, labels])
                    else:
                        if [sentence, labels] not in train_data:
                            train_data.append([sentence, labels])
                else:
                    test_sentence_categories = []
                    for opinions in unprocessed_data[i][1]:
                        dict = opinions.attrib
                        if self.category_label_num[dict['category']] not in test_sentence_categories:
                            test_sentence_categories.append(self.category_label_num[dict['category']])
                    processed_data.append([sentence, test_sentence_categories])

        if is_train:
            print(categories)
            self.categories = categories
            # num_of_valid_data_per_label = [int(num_of_data_per_label[i] * self.validation_percentage) for i in range(len(num_of_data_per_label))]
            # current_num_of_valid_data_per_label = [0] * len(num_of_data_per_label)
            # for data in processed_data:
            #     if current_num_of_valid_data_per_label[int(np.argmax(data[1]))] + 1 > \
            #             num_of_valid_data_per_label[int(np.argmax(data[1]))]:
            #         train_data.append(data)
            #     else:
            #         valid_data.append(data)
            #         current_num_of_valid_data_per_label[int(np.argmax(data[1]))] += 1
            return train_data, categories, valid_data
        else:
            return processed_data


# simple_dataset = SimpleDataset('../datas/ABSA16_Restaurants_Train_SB1_v2.xml', '../datas/EN_REST_SB1_TEST.xml.gold', validation_percentage=2.0)

# word_em_ft = models.KeyedVectors.load_word2vec_format('../fasttext.en.bin', binary=True)
# word_em_glov = models.KeyedVectors.load_word2vec_format('../glove_1.9B_300d.bin', binary=True)
# word_em_w2v = models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
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
        self.sentence_length = 65
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
        entity_label = len(self.simple_dataset.entity) * [0]
        attrib_label = len(self.simple_dataset.attrib) * [0]
        for i in range(len(label)):
            if label[i] == 1:
                entity_label[self.simple_dataset.entity_label_num[self.simple_dataset.category_num_entity[i]]] = 1
                attrib_label[self.simple_dataset.attrib_label_num[self.simple_dataset.category_num_attrib[i]]] = 1
        entity_label = np.array(entity_label)
        attrib_label = np.array(attrib_label)
        return sentence, np.array([label])


if __name__ == '__main__':
    dataset = SimpleDataset('../datas/ABSA16_Restaurants_Train_SB1_v2.xml',
                            '../datas/EN_REST_SB1_TEST.xml.gold', validation_percentage=0.1)
    train_loader = DataLoader(data='train', simple_dataset=dataset)
    valid_loader = DataLoader(data='valid', simple_dataset=dataset)
    test_loader = DataLoader(data='test', simple_dataset=dataset)
    print(train_loader[100])
    # for i in range(len(train_loader)):
    #     print(train_loader[i][0].shape)
        # if 1 not in train_loader[i][1]:
        #     print(train_loader[i])
