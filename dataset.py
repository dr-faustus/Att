from PreProcessing import PreProcessing
import numpy as np
import xml.etree.ElementTree as ET
from gensim import models
from torch.utils.data import dataset
# from allennlp.modules.elmo import Elmo, batch_to_ids
# import allennlp.commands.elmo as E
from tqdm import tqdm
import pickle
from copy import deepcopy


def pair_wise_add(x, y):
    assert len(x) == len(y)
    return [int(x[i] + y[i]) for i in range(len(x))]


def pair_wise_se(x, y):
    assert len(x) == len(y)
    for i in range(len(x)):
        if int(x[i]) > int(y[i]):
            return False
    return True


class SimpleDataset:
    def __init__(self, validation_percentage, dataset_name='sem-2014'):
        self.vocab = []
        self.dataset_name = dataset_name
        self.validation_percentage = validation_percentage
        if dataset_name == 'sem-2016':
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
            self.extract_data('../datas/ABSA16_Restaurants_Train_SB1_v2.xml',
                              '../datas/EN_REST_SB1_TEST.xml.gold')
        elif dataset_name == 'sem-2014':
            self.category_label_num = {
                'service': 0,
                'food': 1,
                'price': 2,
                'ambience': 3,
                'anecdotes/miscellaneous': 4
            }
            self.extract_data('../datas/ABSA14_Restaurants_Train.xml',
                              '../datas/Restaurants_Test_Data_phaseB.xml')

    def extract_data(self, train_file, test_file):
        tree = ET.parse(train_file)
        root = tree.getroot()
        if self.dataset_name == 'sem-2016':
            train_sentences = root.findall('**/sentence')
        elif self.dataset_name == 'sem-2014':
            train_sentences = root.findall('sentence')
        tree = ET.parse(test_file)
        root = tree.getroot()
        if self.dataset_name == 'sem-2016':
            test_sentences = root.findall('**/sentence')
        elif self.dataset_name == 'sem-2014':
            test_sentences = root.findall('sentence')
        self.processed_train_sentences = self.process_data(train_sentences)
        self.processed_test_sentences = self.process_data(test_sentences)
        if self.dataset_name == 'sem-2016':
            self.train_data, self.categories, self.valid_data = self.get_inputs_sem_2016(self.processed_train_sentences,
                                                                                         train_sentences,
                                                                                         is_train=True)
            self.test_data = self.get_inputs_sem_2016(self.processed_test_sentences, test_sentences)
        elif self.dataset_name == 'sem-2014':
            self.train_data, self.categories, self.valid_data = self.get_inputs_sem_2014(self.processed_train_sentences,
                                                                                         train_sentences,
                                                                                         is_train=True)
            self.test_data = self.get_inputs_sem_2014(self.processed_test_sentences, test_sentences)
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

    def get_inputs_sem_2016(self, processed_sentences, unprocessed_data, is_train=False):
        processed_data = []
        categories = []
        num_of_data_per_cat = [0] * len(self.category_label_num.keys())
        for i in range(len(processed_sentences)):
            processed_sentences[i] = processed_sentences[i].split()
        if is_train:
            self.original_sentence = []
            self.valid_original_sentence = []
            self.train_original_sentence = []
        valid_data = []
        train_data = []
        for i in range(len(unprocessed_data)):
            sentence = processed_sentences[i]
            for word in sentence:
                if word not in self.vocab:
                    self.vocab.append(word)
            sentence_attrib = unprocessed_data[i].attrib
            try:
                if sentence_attrib['OutOfScope'] == 'TRUE':
                    continue
            except KeyError:
                pass
            if len(unprocessed_data[i]) > 1:
                if is_train:
                    labels = len(self.category_label_num.keys()) * [0]
                    for opinions in unprocessed_data[i][1]:
                        dict = opinions.attrib
                        if str(dict['category']) not in categories:
                            categories.append(str(dict['category']))
                        labels[self.category_label_num[str(dict['category'])]] = 1
                        num_of_data_per_cat[self.category_label_num[dict['category']]] += 1
                    self.original_sentence.append(unprocessed_data[i][0].text)
                    processed_data.append([sentence, labels])
                else:
                    test_sentence_categories = []
                    labels = len(self.category_label_num.keys()) * [0]
                    for opinions in unprocessed_data[i][1]:
                        dict = opinions.attrib
                        # if self.category_label_num[dict['category']] not in test_sentence_categories:
                        #     test_sentence_categories.append(self.category_label_num[dict['category']])
                        if str(dict['category']) not in categories:
                            categories.append(str(dict['category']))
                        labels[self.category_label_num[str(dict['category'])]] = 1
                    processed_data.append([sentence, labels])
        if is_train:
            num_of_valid_data_per_cat = [int(num_of_data_per_cat[i] * self.validation_percentage) for i in range(len(num_of_data_per_cat))]
            current_num_of_valid_data_per_cat = [0] * len(self.category_label_num.keys())
            for idx, item in enumerate(processed_data):
                sentence = item[0]
                label = item[1]
                temp = pair_wise_add(label, current_num_of_valid_data_per_cat)
                if pair_wise_se(temp, num_of_valid_data_per_cat) is True:
                    valid_data.append([sentence, label])
                    self.valid_original_sentence.append(self.original_sentence[idx])
                    current_num_of_valid_data_per_cat = temp
                else:
                    train_data.append([sentence, label])
                    self.train_original_sentence.append(self.original_sentence[idx])
            return train_data, categories, valid_data
        else:
            return processed_data

    def get_inputs_sem_2014(self, processed_sentences, unprocessed_data, is_train=False):
        processed_data = []
        categories = []
        num_of_data_per_cat = [0] * len(self.category_label_num.keys())
        valid_data = []
        train_data = []
        if is_train:
            self.original_sentence = []
            self.valid_original_sentence = []
            self.train_original_sentence = []
        for i in range(len(processed_sentences)):
            processed_sentences[i] = processed_sentences[i].split()
        for i in range(len(unprocessed_data)):
            sentence = processed_sentences[i]
            if len(unprocessed_data[i]) > 1 and len(unprocessed_data[i][1]) > 0:
                if unprocessed_data[i][1].tag == 'aspectCategories':
                    aspect_cats = unprocessed_data[i][1]
                else:
                    aspect_cats = unprocessed_data[i][2]
                if is_train:
                    labels = 5 * [0]
                    self.original_sentence.append(unprocessed_data[i][0].text)
                    for opinions in aspect_cats:
                        dict = opinions.attrib
                        labels[self.category_label_num[str(dict['category'])]] = 1
                        if dict['category'] not in categories:
                            categories.append(dict['category'])
                        num_of_data_per_cat[self.category_label_num[dict['category']]] += 1
                    processed_data.append([sentence, labels])
                else:
                    test_sentence_categories = []
                    if unprocessed_data[i][1].tag == 'aspectCategories':
                        aspect_cats = unprocessed_data[i][1]
                    else:
                        aspect_cats = unprocessed_data[i][2]
                    for opinions in aspect_cats:
                        dict = opinions.attrib
                        if self.category_label_num[dict['category']] not in test_sentence_categories:
                            test_sentence_categories.append(self.category_label_num[dict['category']])
                    processed_data.append([sentence, test_sentence_categories])
            else:
                if is_train:
                    processed_data.append([sentence, 'NULL'])
                else:
                    processed_data.append([sentence, [self.category_label_num['NULL']]])
        if is_train:
            num_of_valid_data_per_cat = [int(num_of_data_per_cat[i] * self.validation_percentage) for i in
                                         range(len(num_of_data_per_cat))]
            current_num_of_valid_data_per_cat = [0] * len(self.category_label_num.keys())
            for idx, item in enumerate(processed_data):
                sentence = item[0]
                label = item[1]
                temp = pair_wise_add(label, current_num_of_valid_data_per_cat)
                if pair_wise_se(temp, num_of_valid_data_per_cat) is True:
                    valid_data.append([sentence, label])
                    self.valid_original_sentence.append(self.original_sentence[idx])
                    current_num_of_valid_data_per_cat = temp
                else:
                    self.train_original_sentence.append(self.original_sentence[idx])
                    train_data.append([sentence, label])

            return train_data, categories, valid_data
        else:
            return processed_data

    def get_idxed_dataset(self, data):
        if data == 'train':
            train_dataset = []
            for item in self.train_data:
                sentence = [self.vocab.index(item[0][i]) for i in range(len(item[0]))]
                label = item[1]
                train_dataset.append([sentence, label])
            return train_dataset
        elif data == 'test':
            test_dataset = []
            for item in self.test_data:
                sentence = [self.vocab.index(item[0][i]) for i in range(len(item[0]))]
                label = item[1]
                test_dataset.append([sentence, label])
            return test_dataset
        else:
            valid_dataset = []
            for item in self.valid_data:
                sentence = [self.vocab.index(item[0][i]) for i in range(len(item[0]))]
                label = item[1]
                valid_dataset.append([sentence, label])
            return valid_dataset


def get_embeddings(vocab):
    np.random.seed(0)
    word_em = models.KeyedVectors.load_word2vec_format('../yelp_W2V_skipgram.bin', binary=True)
    embeddings = []
    for word in vocab:
        if word in word_em.vocab:
            embeddings.append(word_em[word])
        else:
            embeddings.append(np.random.uniform(-.05, .05, 300))
    embeddings.append(np.random.uniform(-.05, .05, 300))
    embeddings.append(np.zeros(300))
    return np.array(embeddings)


class DataLoader(dataset.Dataset):
    def __init__(self, data='train', simple_dataset=None, dataset_name='sem-2014', padding=True):
        assert simple_dataset is not None
        self.data_type = data
        self.simple_dataset = simple_dataset

        input_data = self.simple_dataset.get_idxed_dataset(data)

        self.sentences = [input_data[i][0] for i in range(len(input_data))]
        self.labels = [input_data[i][1] for i in range(len(input_data))]
        if dataset_name == 'sem-2016':
            self.sentence_length = 66
        elif dataset_name == 'sem-2014':
            self.sentence_length = 33
        self.padding = padding
        self.ending_idx = len(self.simple_dataset.vocab)
        self.padding_idx = len(self.simple_dataset.vocab) + 1
        self.embedding_len = 300

    def __len__(self):
        return len(self.sentences)

    def get_num_of_classes(self):
        return self.simple_dataset.number_of_categories

    def get_sentence_length(self):
        return self.sentence_length

    def __getitem__(self, item):
        sentence = deepcopy(self.sentences[item])
        # if len(sentence) == 0:
        #     sentence.append(self.padding_idx)
        sentence.append(self.ending_idx)
        length = len(sentence)
        if self.padding is True:
            while len(sentence) < self.sentence_length:
                sentence.append(self.padding_idx)
        sentence_rep = np.array(sentence, dtype='long')
        label = np.array(self.labels[item])
        return sentence_rep, np.array([label]), length


def get_data_loaders(validation_percentage=0.1, dataset_name='sem-2014'):
    dataset = SimpleDataset(validation_percentage=validation_percentage, dataset_name=dataset_name)
    train_loader = DataLoader(data='train', simple_dataset=dataset, dataset_name=dataset_name)
    valid_loader = DataLoader(data='valid', simple_dataset=dataset, dataset_name=dataset_name)
    test_loader = DataLoader(data='test', simple_dataset=dataset, dataset_name=dataset_name)
    return train_loader, valid_loader, test_loader, get_embeddings(dataset.vocab)


def return_similar_word_to_vector(vector):
    return word_em.similar_by_vector(vector, topn=20)

