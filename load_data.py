import json
import nltk
import random
import time

# This file is used to load the data from 'Video_Games.json'
# The input 'data_num' is the number of data generated, maximum is 442620,(default=10000)
# Not all data from the file have both the feature 'review text' and 'vote', only 442620 from 2,565,350
# The nltk.word_tokenize() took a lot of time so we first generate data with (str,vote_num) and then choose
# 'data_num' of data from generated data. After that we apply word_tokenize()
# The output data structure should be a list, each element is a tuple like (list_of_tokens,vote_num)


def load_data(data_num=10000):
    start_time = time.time()
    print('loading data...')
    with open('Video_Games.json') as file:
        data = file.read()
        data = data.split('\n')
    print('formatting data...')
    data_labeled = [(json.loads(e)['reviewText'],json.loads(e)['vote'])
                    for e in data if ('"vote"' in e and '"reviewText"' in e)]
    data_labeled = data_labeled[:data_num]
    data_labeled = [(nltk.word_tokenize(review), helpfulness(vote)) for review,vote in data_labeled]
    random.shuffle(data_labeled)
    elapsed_time = time.time() - start_time
    print(len(data_labeled), 'data loaded successfully, Elapsed time:', str(elapsed_time) + 's')
    return data_labeled


def helpfulness(vote):
    vote = int(vote)
    if vote < 3:
        return 1
    elif vote < 5:
        return 2
    elif vote < 7:
        return 3
    elif vote < 11:
        return 4
    else:
        return 5

if __name__ == '__main__':
    data_labeled = load_data()
