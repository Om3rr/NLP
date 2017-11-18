from nltk.corpus import brown
# constants
PERCENTAGE = 10
TAG = 0
COUNTS = 1
FREQ = 1
TUPLE_TAG = 1
TOTAL = 'total'


class BrownCorpus(object):

    def __init__(self, percentage):
        # extract the last PERCENTAGE from brown corpus to be test set
        # the first 100 - PERCENTAGE from brown corpus to be the training set
        brown_news_tagged = brown.tagged_sents(categories='news')
        brown_news_tagged_size = len(brown_news_tagged)
        training_set_size = round(brown_news_tagged_size * percentage / 100)
        self.training_set = brown_news_tagged[-training_set_size:]
        self.test_set = brown_news_tagged[:brown_news_tagged_size - training_set_size]

        # this dict will look like { word1 : {tag1 : count tag 1, tag2 : count tag 2...}}
        self.word_tag_max_dict = {}
        self.words_count = {}

        self.tag_tag_counts_dict = {}

        # initialize the dictionary training_set_word_tag such that for each tuple
        # of word and tag as a key, we will have a value which represents the count
        # (= number of occurrences) of the tag following to the word in all sentences.
        # so, for each pair, if it's the first time we encountered initialize the value
        # to be 1, otherwise, value++
        self.training_set_word_tag = {}
        for sentence in self.training_set:
            for word, tag in sentence:
                if word not in self.training_set_word_tag:
                    self.training_set_word_tag[word] = {}
                    self.words_count[word] = 0
                if tag not in self.training_set_word_tag[word]:
                    self.training_set_word_tag[word][tag] = 0
                self.training_set_word_tag[word][tag] += 1
                self.words_count[word] += 1

            for idx in range(1,len(sentence)):
                # we can change this line of code to set biGram,
                # triGram or whatever we like :D
                tag_1, tag = sentence[idx-1][TUPLE_TAG], sentence[idx][TUPLE_TAG]
                if tag_1 not in self.tag_tag_counts_dict:
                    self.tag_tag_counts_dict[tag_1] = {TOTAL : 0}
                if tag not in self.tag_tag_counts_dict[tag_1]:
                    self.tag_tag_counts_dict[tag_1][tag] = 0
                self.tag_tag_counts_dict[tag_1][tag] += 1
                self.tag_tag_counts_dict[tag_1][TOTAL] += 1

    def calc_test_set_error_rate(self):
        misses = 0
        tries = 0
        # iterating over (w,t) pairs in test set. foreach word check if the max
        # tag equals to the tagged word.
        # if it is count is as 'hit' otherwise as a 'miss' in the end we will
        # calculate the miss rate as (total misses / total shots)
        for sentence in self.test_set:
            for w,t in sentence:
                misses += 0 if t == self.word_tag_max_dict[w][TAG] else 1
                tries += 1
        return misses / tries

    def get_max_tag(self, word):
        """
        :param word: word to check for the most common tag.
        :return: the tag that maximize P(tag|word)
        """
        if word not in self.training_set_word_tag:
            return 'NN'
        tags_pressed = sorted(list(self.training_set_word_tag[word].items()),
                            key=lambda x: x[1], reverse=True)
        self.word_tag_max_dict[word] = \
            tags_pressed[0][TAG]

    def emission(self, word, tag):
        # this function will calculate P(tag | word)
        if tag not in self.training_set_word_tag[word]:
            return 0
        return self.training_set_word_tag[word][tag] / self.words_count[word]

    def transition(self, tag, tag_1):
        if tag not in self.tag_tag_counts_dict[tag_1]:
            return 0
        return self.tag_tag_counts_dict[tag_1][tag] / \
               self.tag_tag_counts_dict[tag_1][TOTAL]


def main():
    # initialize brown corpus training set and test set, test data will be the last
    # PERCENTAGE
    bc = BrownCorpus(PERCENTAGE)

    # return a list such that for each word we will have the most common tag and the
    # probability of p(tag|word)
    # bc.get_list_most_suitable_tag_word()

main()

