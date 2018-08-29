from plotting import *
from dataset import get_data_loaders
from dataset import SimpleDataset
from dataset import DataLoader
from model import get_sentence_weights
import seaborn as sns
import string
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt

fontP = FontProperties()
fontP.set_size('small')

plt.switch_backend('agg')
# for test
validation_percentage = 0.1


def validate(learning_rate, batch_size, dataset_name, model_type):
    train_loader, validation_loader, test_loader = get_data_loaders(validation_percentage, dataset_name)
    net = Net(250, train_loader, test_loader, validation_loader, learning_rate, model_type=model_type)
    train_loss, test_loss = net.train(batch_size=batch_size, validate=True)


def plot_validations():
    learning_rate_validation(300, dataset_name, learning_rate_list, batch_size, model_type,
                             num_of_topics, hidden_size, topic_hidden_size, drop_out_prob)
    batch_size_validation_plot(300, dataset_name, learning_rate, batch_size_list, model_type,
                               num_of_topics, hidden_size, topic_hidden_size, drop_out_prob)
    drop_out_porb_validation(300, dataset_name, learning_rate, batch_size, model_type,
                             num_of_topics, hidden_size, topic_hidden_size, drop_out_prob_list)
    num_of_topics_validation(300, dataset_name, learning_rate, batch_size, model_type, num_of_topics_list, hidden_size,
                             topic_hidden_size, drop_out_prob)
    hidden_size_validation(300, dataset_name, learning_rate, batch_size, model_type, num_of_topics, hidden_size_list,
                           topic_hidden_size, drop_out_prob)
    topic_hidden_size_validation(300, dataset_name, learning_rate, batch_size, model_type,
                                 num_of_topics, hidden_size, topic_hidden_size_list, drop_out_prob)


def test(learning_rate, batch_size, dataset_name, model_type, early_stopping_mode, early_stopping_min_delta, early_stopping_patience):
    train_loader, validation_loader, test_loader = get_data_loaders(validation_percentage, dataset_name)
    net = Net(300, train_loader, test_loader, validation_loader, learning_rate, model_type,
              early_stopping_mode, early_stopping_min_delta, early_stopping_patience,
              input_size=input_size, num_of_topics=num_of_topics, hidden_size=hidden_size,
              topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob)
    result = net.train(batch_size=batch_size, validate=False)
    print(result)


def identity(x):
    return x


def sentence_weight_examine(sentence_idx):
    dataset = SimpleDataset(validation_percentage=validation_percentage, dataset_name=dataset_name)
    loader = DataLoader(data='valid', simple_dataset=dataset, dataset_name='sem-2016', padding=False)
    # for idx, sentence in enumerate(dataset.valid_original_sentence):
    #     print(idx)
    #     print(sentence)
    #     print(dataset.valid_data[idx][0])
    #     print(loader[idx][1])
    # f, ax = plt.subplots(figsize=(9, 6))
    # flights = flights_long.pivot("month", "year", "passengers")
    # print(type(flights))
    # exit()
    item = loader[sentence_idx][0]
    preprocessed_sentence = dataset.valid_data[sentence_idx][0]
    orig_sentence = dataset.valid_original_sentence[sentence_idx]
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    orig_sentence = orig_sentence.translate(translator)
    orig_sentence = orig_sentence.split()
    weights, existence = get_sentence_weights('./topic-attention', item)

    topics = []
    for i in range(11):
        topics.append(str(i + 1))
    stopwords_english = set(stopwords.words('english'))
    attention_weights = []
    words = []
    for idx, weight in enumerate(weights):
        weight = list(weight.squeeze(0).squeeze(-1).detach().numpy())
        temp = []
        for word in orig_sentence:
            words.append(word)
            if word.lower() in stopwords_english:
                temp.append(float(0.0))
            else:
                temp.append(float(weight[preprocessed_sentence.index(word.lower())]))
        attention_weights.append(np.array(temp))
    attention_weights = np.array(attention_weights)
    attention_weights = attention_weights.transpose()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(attention_weights, annot=True, fmt=".2f", ax=ax, xticklabels=topics, yticklabels=orig_sentence,
                cmap="YlGnBu", cbar_kws={'label': 'Attention Weights'})
    plt.xlabel('Topics')
    plt.ylabel('Sentence')
    plt.savefig('./attention_heatmap/valid_sentence_#' + str(sentence_idx) + '_attention_weights')

    topic_probs = []
    for idx in range(len(existence)):
        prob = float(existence[idx])
        topic_probs.append(prob)
    topic_probs = [np.array(topic_probs)]
    topic_probs = np.array(topic_probs).transpose()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(topic_probs, annot=True, fmt=".2f", ax=ax, xticklabels=['Topic Probabilities'], yticklabels=topics,
                cmap="YlGnBu")
    plt.savefig('./attention_heatmap/valid_sentence_#' + str(sentence_idx) + '_topic_probs')

    # print(list(weights))
    # print(list(existence))


num_of_topics_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
hidden_size_list = [16, 32, 64, 128, 256]
topic_hidden_size_list = [4, 8, 16, 32, 64, 128]
drop_out_prob_list = [0.2, 0.4, 0.5, 0.6, 0.7]


batch_size_list = [16, 32, 64, 128, 256]
learning_rate_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]

learning_rate = 0.001

input_size = 300

early_stopping_mode = 'min'
early_stopping_min_delta = 0

model_type = 'topic-attention'
dataset_name = 'sem-2014'
num_of_topics = 6
hidden_size = 128
topic_hidden_size = 16
drop_out_prob = 0.6
batch_size = 128
early_stopping_patience = 20
test(learning_rate, batch_size, dataset_name, model_type, early_stopping_mode, early_stopping_min_delta, early_stopping_patience)

for i in range(300):
    sentence_weight_examine(i)
exit()

model_type = 'vanilla-attention'
dataset_name = 'sem-2016'
num_of_topics = 11
hidden_size = 128
topic_hidden_size = 32
drop_out_prob = 0.6
batch_size = 128
early_stopping_patience = 20
test(learning_rate, batch_size, dataset_name, model_type, early_stopping_mode, early_stopping_min_delta, early_stopping_patience)

dataset_name = 'sem-2014'
num_of_topics = 6
hidden_size = 128
topic_hidden_size = 16
drop_out_prob = 0.6
batch_size = 128
learning_rate = 0.001
early_stopping_patience = 20
test(learning_rate, batch_size, dataset_name, model_type, early_stopping_mode, early_stopping_min_delta, early_stopping_patience)

model_type = 'topic-attention-without-squash'
dataset_name = 'sem-2016'
num_of_topics = 11
hidden_size = 128
topic_hidden_size = 32
drop_out_prob = 0.6
batch_size = 128
early_stopping_patience = 20
test(learning_rate, batch_size, dataset_name, model_type, early_stopping_mode, early_stopping_min_delta, early_stopping_patience)

dataset_name = 'sem-2014'
num_of_topics = 6
hidden_size = 128
topic_hidden_size = 16
drop_out_prob = 0.6
batch_size = 128
learning_rate = 0.001
early_stopping_patience = 20
test(learning_rate, batch_size, dataset_name, model_type, early_stopping_mode, early_stopping_min_delta, early_stopping_patience)

model_type = 'topic-attention'
dataset_name = 'sem-2016'
num_of_topics = 11
hidden_size = 128
topic_hidden_size = 32
drop_out_prob = 0.6
batch_size = 128
early_stopping_patience = 20
test(learning_rate, batch_size, dataset_name, model_type, early_stopping_mode, early_stopping_min_delta, early_stopping_patience)

dataset_name = 'sem-2014'
num_of_topics = 6
hidden_size = 128
topic_hidden_size = 16
drop_out_prob = 0.6
batch_size = 128
learning_rate = 0.001
early_stopping_patience = 20
test(learning_rate, batch_size, dataset_name, model_type, early_stopping_mode, early_stopping_min_delta, early_stopping_patience)