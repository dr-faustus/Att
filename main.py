from plotting import *
from dataset import get_data_loaders

fontP = FontProperties()
fontP.set_size('small')

plt.switch_backend('agg')


def validate(learning_rate, batch_size, dataset_name, model_type):
    train_loader, validation_loader, test_loader = get_data_loaders(0.1, dataset_name)
    net = Net(250, train_loader, test_loader, validation_loader, learning_rate, model_type=model_type)
    train_loss, test_loss = net.train(batch_size=batch_size, validate=True)


def test(learning_rate, batch_size, dataset_name, model_type):
    train_loader, validation_loader, test_loader = get_data_loaders(0.0, dataset_name)
    net = Net(300, train_loader, test_loader, validation_loader, learning_rate, input_size=input_size,
              num_of_topics=num_of_topics, hidden_size=hidden_size,
              topic_hidden_size=topic_hidden_size, drop_out_prob=drop_out_prob, model_type=model_type)
    result = net.train(batch_size=batch_size, validate=False)
    print(result)


batch_size_list = [16, 32, 64, 128, 256]
num_of_topics_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
learning_rate_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
hidden_size_list = [50, 100, 150, 200, 300]
context_size_list = [100, 200, 300, 500]
topic_hidden_size_list = [5, 10, 20, 50, 100]
drop_out_prob_list = [0.2, 0.4, 0.5, 0.6, 0.7]

learning_rate = 0.001
batch_size = 128

num_of_topics = 5
hidden_size = 150
input_size = 300
classification_size = 5
context_size = 300
topic_hidden_size = 20
drop_out_prob = 0.6

dataset_name = 'sem-2014'
model_type = 'topic-attention'

num_of_topics_validation(300, dataset_name, learning_rate, batch_size, model_type,
                         num_of_topics_list, hidden_size, topic_hidden_size, drop_out_prob)
dataset_name = 'sem-2016'
num_of_topics_validation(300, dataset_name, learning_rate, batch_size, model_type,
                         num_of_topics_list, hidden_size, topic_hidden_size, drop_out_prob)

# test(learning_rate, batch_size, dataset_name, model_type)
