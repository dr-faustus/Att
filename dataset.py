from PreProcessing import PreProcessing
import numpy as np
import xml.etree.ElementTree as ET
from gensim import models
from torch.utils.data import dataset
from allennlp.modules.elmo import Elmo, batch_to_ids
import allennlp.commands.elmo as E
from tqdm import tqdm
import pickle


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
                    labels = len(self.category_label_num.keys()) * [0]
                    for opinions in unprocessed_data[i][1]:
                        dict = opinions.attrib
                        if str(dict['category']) not in categories:
                            categories.append(str(dict['category']))
                        labels[self.category_label_num[str(dict['category'])]] = 1.0
                        num_of_data_per_cat[self.category_label_num[dict['category']]] += 1
                    processed_data.append([sentence, labels])
                else:
                    test_sentence_categories = []
                    for opinions in unprocessed_data[i][1]:
                        dict = opinions.attrib
                        if self.category_label_num[dict['category']] not in test_sentence_categories:
                            test_sentence_categories.append(self.category_label_num[dict['category']])
                    processed_data.append([sentence, test_sentence_categories])
        if is_train:
            num_of_valid_data_per_cat = [int(num_of_data_per_cat[i] * self.validation_percentage) for i in range(len(num_of_data_per_cat))]
            current_num_of_valid_data_per_cat = [0] * len(self.category_label_num.keys())
            for item in processed_data:
                sentence = item[0]
                label = item[1]
                temp = pair_wise_add(label, current_num_of_valid_data_per_cat)
                if pair_wise_se(temp, num_of_valid_data_per_cat) is True:
                    valid_data.append([sentence, label])
                    current_num_of_valid_data_per_cat = temp
                else:
                    train_data.append([sentence, label])
            return train_data, categories, valid_data
        else:
            return processed_data

    def get_inputs_sem_2014(self, processed_sentences, unprocessed_data, is_train=False):
        processed_data = []
        categories = []
        num_of_data_per_cat = [0] * len(self.category_label_num.keys())
        valid_data = []
        train_data = []
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
            for item in processed_data:
                sentence = item[0]
                label = item[1]
                temp = pair_wise_add(label, current_num_of_valid_data_per_cat)
                if pair_wise_se(temp, num_of_valid_data_per_cat) is True:
                    valid_data.append([sentence, label])
                    current_num_of_valid_data_per_cat = temp
                else:
                    train_data.append([sentence, label])
            return train_data, categories, valid_data
        else:
            return processed_data


word_em = models.KeyedVectors.load_word2vec_format('../yelp_W2V_skipgram.bin', binary=True)
# word_em = models.KeyedVectors.load_word2vec_format('../glove_1.9B_300d.bin', binary=True)
# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = '../elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
# elmo = E.ElmoEmbedder(options_file, weight_file)
simple_dataset = None


# def save_elmo_emb(sentences, data_type, dataset_name):
#     temp = []
#     for sentence in tqdm(sentences):
#         sentence_ = elmo.embed_sentence(sentence)
#         temp.append(sentence_)
#     sentences = temp
#     with open(data_type + '_elmo_' + dataset_name, "wb") as fp:
#         pickle.dump(sentences, fp)
#
#
# def load_elmo_emb(data_type, dataset_name):
#     with open(data_type + '_elmo_' + dataset_name, 'rb') as fp:
#         sentences = pickle.load(fp)
#     return sentences


class DataLoader(dataset.Dataset):
    def __init__(self, data='train', simple_dataset=None, dataset_name='sem-2014', padding=True):
        assert simple_dataset is not None
        self.data_type = data
        self.simple_dataset = simple_dataset
        if data == 'train':
            input_data = self.simple_dataset.train_data
        elif data == 'test':
            input_data = self.simple_dataset.test_data
        else:
            input_data = self.simple_dataset.valid_data

        self.sentences = [input_data[i][0] for i in range(len(input_data))]
        self.labels = [input_data[i][1] for i in range(len(input_data))]
        if dataset_name == 'sem-2016':
            self.sentence_length = 65
        elif dataset_name == 'sem-2014':
            self.sentence_length = 33
        self.padding = padding
        self.embedding_len = 300

    def __len__(self):
        return len(self.sentences)

    def get_num_of_classes(self):
        return self.simple_dataset.number_of_categories

    def get_sentence_length(self):
        return self.sentence_length

    def __getitem__(self, item):
        sentence = self.sentences[item]
        sentence_rep = []
        for word in sentence:
            try:
                sentence_rep.append(word_em[word])
            except KeyError:
                continue
        if self.padding is True:
            while len(sentence_rep) < self.sentence_length:
                sentence_rep.append(np.array(np.zeros(self.embedding_len), dtype='float32'))
        sentence_rep = np.array(sentence_rep, dtype='float64')
        label = np.array(self.labels[item])
        return sentence_rep, np.array([label])


def get_data_loaders(validation_percentage=0.1, dataset_name='sem-2014'):
    dataset = SimpleDataset(validation_percentage=validation_percentage, dataset_name=dataset_name)
    train_loader = DataLoader(data='train', simple_dataset=dataset, dataset_name=dataset_name)
    valid_loader = DataLoader(data='valid', simple_dataset=dataset, dataset_name=dataset_name)
    test_loader = DataLoader(data='test', simple_dataset=dataset, dataset_name=dataset_name)
    return train_loader, valid_loader, test_loader


def return_similar_word_to_vector(vector):
    return word_em.similar_by_vector(vector, topn=20)


if __name__ == '__main__':
    train, valid, test = get_data_loaders(validation_percentage=0.1, dataset_name='sem-2016')
    print(train[0])
    # for i in range(len(train_loader)):
    #     print(train_loader[i][0].shape)
        # if 1 not in train_loader[i][1]:
        #     print(train_loader[i])
