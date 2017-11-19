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
            sentence += [('','STOP')]
            for idx in range(1, len(sentence)):
                prev_tag, tag = sentence[idx-1][TUPLE_TAG], sentence[idx][TUPLE_TAG]
                if prev_tag not in self.tag_tag_counts_dict:
                    self.tag_tag_counts_dict[prev_tag] = {}
                if tag not in self.tag_tag_counts_dict[prev_tag]:
                    self.tag_tag_counts_dict[prev_tag][tag] = 0
                self.tag_tag_counts_dict[prev_tag][tag] += 1




        self.viterbiTable = {}


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
        if tag not in self.training_set_tag_word:
            return 0
        if word not in self.training_set_tag_word[tag]:
            return 0

        return self.training_set_tag_word[tag][word] / self.tags_count[tag]

    def emission_add_1_smoothing(self, word, tag):
        # this function will calculate p add_1(word | tag)
        if tag not in self.training_set_tag_word:
            return 0
        if word not in self.training_set_tag_word[tag]:
            return 0
        num_of_tags = len(self.tags_count)
        return (self.training_set_tag_word[tag][word] + 1) / \
               (self.tags_count[tag] + num_of_tags)

    def transition(self, prev_tag, tag):
        if prev_tag not in self.tag_tag_counts_dict:
            return 0
        if tag not in self.tag_tag_counts_dict[prev_tag]:
            return 0
        return self.tag_tag_counts_dict[prev_tag][tag] / \
               self.tags_count[prev_tag]


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

    def r(self,k, words, tags):
        mult = 1
        for idx in range(1,k):
            mult *= self.transition(tags[idx-1], tags[idx])
            mult *= self.emission(words[idx], tags[idx])
        return mult

    def pi(self,words, k, v):
        if k == 0:
            return 1, '*'
        if (k,v) in self.viterbiTable:
            return self.viterbiTable[(k,v)]
        w_freq, w = max([(self.pi(words, k-1,w)[0] * self.transition(v,w) * self.emission(words[k], v)
                          , w)
                         for w in self.tag_tag_counts_dict],
                    key=lambda x:x[0])
        self.viterbiTable[(k,v)] = (w_freq, w)
        return w_freq, w

    def viterbi(self, words):
        tags = []
        words = ["*"] + words.split(" ")
        _,last_tag = max([(self.pi(words, len(words) - 1, w)[0] * self.transition(w, 'STOP') , w)
                          for w in self.tag_tag_counts_dict.keys()], key=lambda x:x[0])
        tags.append(last_tag)
        for idx in range(len(words) - 1,1,-1):
            last_tag = self.pi(words, idx, last_tag)
            tags.append(last_tag[1])

        return tags[::-1]

    def print_training_tag_word_dict(self):
        print(self.training_set_tag_word)

    def print_training_word_tag_dict(self):
        print(self.training_set_word_tag)

    def print_tags_count(self):

        print(sorted(list(self.tags_count.items()),
                                  key=lambda x: x[1], reverse=True))

    def print_words_count(self):
        print(sorted(list(self.words_count.items()),
                                  key=lambda x: x[1], reverse=True))



def main():
    # initialize brown corpus training set and test set, test data will be the last
    # PERCENTAGE
    bc = BrownCorpus(PERCENTAGE)

    # return a list such that for each word we will have the most common tag and the
    # probability of p(tag|word)
    # bc.get_list_most_suitable_tag_word()

    bc.print_training_tag_word_dict()
    bc.print_training_word_tag_dict()

    print(bc.emission('Nothing', 'PN-HL'))

    print(bc.emission_add_1_smoothing('Nothing', 'PN-HL'))


main()

