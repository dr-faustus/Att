from model import Net
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import gc
from dataset import get_data_loaders
import pickle
from matplotlib.pyplot import figure

fontP = FontProperties()
fontP.set_size('small')

plt.switch_backend('agg')

early_stopping_mode = 'min'
early_stopping_min_delta = 0
early_stopping_patience = 10


def batch_size_validation_plot(num_of_epochs, dataset_name, learning_rate, batch_list, model_type,
                               num_of_topics, hidden_size, topic_hidden_size, drop_out_prob):
    result = dict()
    dump_file_name = './val_results/batch_size_valid_result_' + model_type + '_' + dataset_name
    train_loader, validation_loader, test_loader = get_data_loaders(0.1, dataset_name=dataset_name)
    for batch_size in batch_list:
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate, model_type,
                  early_stopping_mode, early_stopping_min_delta, early_stopping_patience,
                  num_of_topics=num_of_topics, hidden_size=hidden_size, input_size=300,
                  topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob)
        gc.collect()
        train_loss, validation_loss = net.train(batch_size=batch_size, validate=True)
        result[batch_size] = [train_loss, validation_loss]
    with open(dump_file_name, 'wb') as fp:
        pickle.dump(result, fp)


def num_of_topics_validation(num_of_epochs, dataset_name, learning_rate, batch_size, model_type,
                             num_of_topics_list, hidden_size, topic_hidden_size, drop_out_prob):
    result = dict()
    dump_file_name = './val_results/num_of_topics_valid_result_' + model_type + '_' + dataset_name
    train_loader, validation_loader, test_loader = get_data_loaders(0.1, dataset_name=dataset_name)
    for num_of_topics in num_of_topics_list:
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate, model_type,
                  early_stopping_mode, early_stopping_min_delta, early_stopping_patience,
                  num_of_topics=num_of_topics, hidden_size=hidden_size, input_size=300,
                  topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob)
        gc.collect()
        train_loss, validation_loss = net.train(batch_size=batch_size, validate=True)
        result[num_of_topics] = [train_loss, validation_loss]
    with open(dump_file_name, 'wb') as fp:
        pickle.dump(result, fp)


def learning_rate_validation(num_of_epochs, dataset_name, learning_rate_list, batch_size, model_type,
                             num_of_topics, hidden_size, topic_hidden_size, drop_out_prob):
    result = dict()
    dump_file_name = './val_results/learning_rate_valid_result_' + model_type + '_' + dataset_name
    train_loader, validation_loader, test_loader = get_data_loaders(0.1, dataset_name=dataset_name)
    for learning_rate in learning_rate_list:
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate, model_type,
                  early_stopping_mode, early_stopping_min_delta, early_stopping_patience,
                  num_of_topics=num_of_topics, hidden_size=hidden_size, input_size=300,
                  topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob)
        gc.collect()
        train_loss, validation_loss = net.train(batch_size=batch_size, validate=True)
        result[learning_rate] = [train_loss, validation_loss]
    with open(dump_file_name, 'wb') as fp:
        pickle.dump(result, fp)


def hidden_size_validation(num_of_epochs, dataset_name, learning_rate, batch_size, model_type,
                           num_of_topics, hidden_size_list, topic_hidden_size, drop_out_prob):
    result = dict()
    dump_file_name = './val_results/hidden_size_valid_result_' + model_type + '_' + dataset_name
    train_loader, validation_loader, test_loader = get_data_loaders(0.1, dataset_name=dataset_name)
    for hidden_size in hidden_size_list:
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate, model_type,
                  early_stopping_mode, early_stopping_min_delta, early_stopping_patience,
                  num_of_topics=num_of_topics, hidden_size=hidden_size, input_size=300,
                  topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob)
        gc.collect()
        train_loss, validation_loss = net.train(batch_size=batch_size, validate=True)
        result[hidden_size] = [train_loss, validation_loss]
    with open(dump_file_name, 'wb') as fp:
        pickle.dump(result, fp)


def topic_hidden_size_validation(num_of_epochs, dataset_name, learning_rate, batch_size, model_type,
                                 num_of_topics, hidden_size, topic_hidden_size_list, drop_out_prob):
    result = dict()
    dump_file_name = './val_results/topic_hidden_size_valid_result_' + model_type + '_' + dataset_name
    train_loader, validation_loader, test_loader = get_data_loaders(0.1, dataset_name=dataset_name)
    for topic_hidden_size in topic_hidden_size_list:
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate, model_type,
                  early_stopping_mode, early_stopping_min_delta, early_stopping_patience,
                  num_of_topics=num_of_topics, hidden_size=hidden_size, input_size=300,
                  topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob)
        gc.collect()
        train_loss, validation_loss = net.train(batch_size=batch_size, validate=True)
        result[topic_hidden_size] = [train_loss, validation_loss]
    with open(dump_file_name, 'wb') as fp:
        pickle.dump(result, fp)


def drop_out_porb_validation(num_of_epochs, dataset_name, learning_rate, batch_size, model_type,
                             num_of_topics, hidden_size, topic_hidden_size, drop_out_prob_list):
    result = dict()
    dump_file_name = './val_results/drop_out_valid_result_' + model_type + '_' + dataset_name
    train_loader, validation_loader, test_loader = get_data_loaders(0.1, dataset_name=dataset_name)
    for drop_out_prob in drop_out_prob_list:
        net = Net(num_of_epochs, train_loader, test_loader, validation_loader, learning_rate, model_type,
                  early_stopping_mode, early_stopping_min_delta, early_stopping_patience,
                  num_of_topics=num_of_topics, hidden_size=hidden_size, input_size=300,
                  topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob)
        gc.collect()
        train_loss, validation_loss = net.train(batch_size=batch_size, validate=True)
        result[drop_out_prob] = [train_loss, validation_loss]
    with open(dump_file_name, 'wb') as fp:
        pickle.dump(result, fp)


def read_plot():
    src = './val_results/drop_out_valid_result_topic-attention_sem-2016'
    with open(src, 'rb') as fp:
        result = pickle.load(fp)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8, forward=True)
    num_of_epoches = 100

    num_of_topics_list = []
    valid_loss_list = []

    for num_of_topics in result.keys():
        # print(result[num_of_topics][0])
        # if num_of_topics == 1 or num_of_topics == 4 or num_of_topics == 7:
        ax.plot(np.arange(1, num_of_epoches + 1, 1), result[num_of_topics][1][:num_of_epoches], '-', label=str(num_of_topics) + ' valid loss')
        # avr_loss = sum(result[num_of_topics][1][:num_of_epoches]) / num_of_epoches
        # num_of_topics_list.append(num_of_topics)
        # valid_loss_list.append(avr_loss)
        # print(str(num_of_topics) + ': ' + str(avr_loss))
    # plt.plot(num_of_topics_list, valid_loss_list, 'k', label='valid_loss_list')
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large', prop=fontP)
    legend.get_frame().set_facecolor('#FFFFFF')
    plt.savefig('temp')


if __name__ == '__main__':
    read_plot()