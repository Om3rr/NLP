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
        self.test_set = brown_news_tagged[-training_set_size:]
        self.training_set = brown_news_tagged[:brown_news_tagged_size - training_set_size]

        self.words_count = {}
        self.tags_count = {}
        self.max_tags = {}

        self.tag_tag_counts_dict = {}

        # initialize the dictionary training_set_word_tag such that for each tuple
        # of word and tag as a key, we will have a value which represents the count
        # (= number of occurrences) of the tag following to the word in all sentences.
        # so, for each pair, if it's the first time we encountered initialize the value
        # to be 1, otherwise, value++
        self.training_set_word_tag = {}

        self.training_set_tag_word = {}

        for sentence in self.training_set:
            for word, tag in sentence:
                if word not in self.training_set_word_tag:
                    self.training_set_word_tag[word] = {}
                    self.words_count[word] = 0
                if tag not in self.training_set_word_tag[word]:
                    self.training_set_word_tag[word][tag] = 0
                self.training_set_word_tag[word][tag] += 1
                self.words_count[word] += 1

        for sentence in self.training_set:
            for word, tag in sentence:
                if tag not in self.training_set_tag_word:
                    self.training_set_tag_word[tag] = {}
                    self.tags_count[tag] = 0
                if word not in self.training_set_tag_word[tag]:
                    self.training_set_tag_word[tag][word] = 0
                self.training_set_tag_word[tag][word] += 1
                self.tags_count[tag] += 1

        # we can change this line of code to set biGram,
        # triGram or whatever we like :D
        for sentence in self.training_set:
            for idx in range(1, len(sentence)):
                prev_tag, tag = sentence[idx-1][TUPLE_TAG], sentence[idx][TUPLE_TAG]
                if prev_tag not in self.tag_tag_counts_dict:
                    self.tag_tag_counts_dict[prev_tag] = {}
                if tag not in self.tag_tag_counts_dict[prev_tag]:
                    self.tag_tag_counts_dict[prev_tag][tag] = 0
                self.tag_tag_counts_dict[prev_tag][tag] += 1



        print(self.calculate_errors())


    def get_max_tag(self, word):
        """
        :param word: word to check for the most common tag.
        :return: the tag that maximize P(tag|word)
        """
        if word not in self.training_set_word_tag:
            return 'NN'
        if word not in self.max_tags:
            tags_pressed = sorted(list(self.training_set_word_tag[word].items()),
                                  key=lambda x: x[1], reverse=True)
            self.max_tags[word] = \
                tags_pressed[0][TAG]
        return self.max_tags[word]


    def emission(self, word, tag):
        # this function will calculate p (word | tag)
        if word not in self.training_set_tag_word[tag]:
            return 0
        return self.training_set_tag_word[tag][word] / self.tags_count[tag]

    def transition(self, prev_tag, tag):
        if prev_tag not in self.tag_tag_counts_dict:
            return 0
        if tag not in self.tag_tag_counts_dict[prev_tag]:
            return 0
        return self.tag_tag_counts_dict[prev_tag][tag] / \
               self.tags_count[prev_tag]

    def print_training_set_word_tag(self):
        for word, tag in self.training_set_word_tag.items():
            #print(word, self.get_max_tag(word))
            print(word, tag)

    def calculate_errors(self):
        """
        this function calculate the training, test and total model errors.
        and return it as a tuple
        """
        ## training error
        training_misses, training_words = 0,0
        for sentence in self.training_set:
            for word, tag in sentence:
                training_misses += 1 if tag != self.get_max_tag(word) else 0
                training_words += 1

        ## test error
        test_misses, test_words = 0, 0
        for sentence in self.test_set:
            for word, tag in sentence:
                test_misses += 1 if tag != self.get_max_tag(word) else 0
                test_words += 1

        ## total error
        total_error = (training_misses + test_misses) / (training_words + test_words)


        return training_misses / training_words, test_misses / test_words, total_error




def main():
    # initialize brown corpus training set and test set, test data will be the last
    # PERCENTAGE
    bc = BrownCorpus(PERCENTAGE)

    # return a list such that for each word we will have the most common tag and the
    # probability of p(tag|word)
    # bc.get_list_most_suitable_tag_word()

main()

