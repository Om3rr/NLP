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

        self.known_words = set()
        self.test_words = set()
        self.unknown_words = set()

        self.tags = set()

        for sentence in self.training_set:
            for word, tag in sentence:
                self.known_words.add(word)
                self.tags.add(tag)



        for sentence in self.test_set:
            for word, tag in sentence:
                self.test_words.add(word)

        # set of unknown words = test words Minus known words
        self.unknown_words = self.test_words - self.known_words

        print(self.training_set)
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
        self.prob = {}
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
        """
        :param word:
        :param tag:
        :return: p(word | tag) = count(tag, word) / count(tag)
        """
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
        """
        :param prev_tag:
        :param tag:
        :return: count(w, v) / count(w) = q(v|w)
        """
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
        # known words error rate
        known_words_misses, known_words = 0,0
        unknown_words_misses, unknown_words = 0, 0
        for sentence in self.test_set + self.training_set:
            for word, tag in sentence:
                if word in self.training_set_word_tag:
                    known_words_misses += 1 if tag != self.get_max_tag(word) else 0
                    known_words += 1
                else:
                    unknown_words_misses += 1 if tag != 'NN' else 0
                    unknown_words += 1

        # total error
        total_error = (known_words_misses + unknown_words_misses) / \
                      (known_words + unknown_words)

        return known_words_misses / known_words, unknown_words_misses / unknown_words, \
               total_error

    def r(self,k, words, tags):
        mult = 1
        for idx in range(1,k):
            mult *= self.transition(tags[idx-1], tags[idx])
            mult *= self.emission(words[idx], tags[idx])
        return mult

    def pi(self,words, k, v):
        if k == 0 and v == '*':
            return 1, '*'
        if (k,v) in self.viterbiTable:
            return self.viterbiTable[(k,v)]
        w_freq, w = max([(self.pi(words, k-1,w)[0] * self.transition(v,w) * self.emission(words[k], v)
                          , w)
                         for w in self.tag_tag_counts_dict],
                    key=lambda x:x[0])
        self.viterbiTable[(k,v)] = (w_freq, w)
        return w_freq, w

    def viterbi(self, stentence):
        tags = []
        stentence = ["*"] + stentence.split(" ")
        _,last_tag = max([(self.pi(stentence, len(stentence) - 1, w)[0] * self.transition(w, 'STOP') , w)
                          for w in self.tag_tag_counts_dict.keys()], key=lambda x:x[0])
        tags.append(last_tag)
        for idx in range(len(stentence)-1, 1, -1):
            last_tag = self.pi(stentence, idx, last_tag)
            tags.append(last_tag[1])

        return tags[::-1]




    def viterbi2(self, sentence):
        """

        Viterbi method gets a sentence (x1,x2,...,xn)
        :param self:
        :param sentence: x1, x2, x3 ...., xn
        :return:
        1. The max probability of tags to this sentence
        2. The tags themselves with the highest probabilities to this sentence
        """
        def find_set(k):
            """
            This Method gives us Sk = optional tags at position k
            :param k: the index
            :return: the optional tags at this position
            """
            # for any k in the length of the sentence we will return S= all tags
            # for k == 0 we will return '*'
            if k in range(1, len(sentence)+1):
                return self.tags
            elif k == 0:
                return {'*'}

        def pi2(k, v):
            """
            :param k: the (word) position in the sentence
            :param v: the last tag in the kth position
            :return: max probability of tags sequence ending in tag v at position k
            """
            prob = {}
            # initialization set pi(0,*) = 1
            if k == 0 and v == '*':
                return 1., '*'

            else:
                # w belongs to S_k-1
                for w in find_set(k-1):
                    # pi(k-1,w) * q(v|w) * e(x_k|v)
                    prev = pi2(k-1, w)[0]
                    transition = self.transition(v, w)
                    emission = self.emission(sentence[k-1].lower(), v)
                    probability = prev * transition * emission
                    prob[tuple((w, v))] = probability
                # max according to the probabilities
                max_tuple = max(prob.items(), key=lambda x: x[1])
                # max_tuple[1] = prob1
                # max_tuple[0][0] = the word of the first {(w,v), prob1} = w1
                return max_tuple[1], max_tuple[0][0]

        # split the sentence according to spaces
        sentence = sentence.split(" ")
        sentence = ["START"] + sentence
        n = len(sentence)
        tags = {}
        bp = {}

        # for k=1,2...,n
        for k in range(1, n+1):
            prob = {}
            # v belongs to S_k for k belongs to {1,2,3...,k}
            for v in find_set(k):
                value, w = pi2(k, v)
                if k == n:
                    value *= self.transition("STOP", v)
                prob[tuple((k, v))] = value
                bp[tuple((k, v))] = w
            max_tuple = max(prob.items(), key=lambda x: x[1])
            # bp (k, v)= tag w
            bp[tuple((k, max_tuple[0][-1]))] = max_tuple[0][1]
        tags[n] = max_tuple[0][1]
        print(sorted(list(bp.items()),
                                  key=lambda x: x[1], reverse=True))
        # for k = (n-1)....1
        # tags[k] = bp(k+1, tags[k+1])

        for k in range(n-1, 0, -1):
            print(k)
            tags[k] = bp[tuple((k+1, tags[k+1]))]

        # return tag_list = tags[1],....tags[n]
        tag_list = []
        n = len(tags)
        for i in range(1, n + 1):
            tag_list.append(tags[i])
        print(max_tuple)
        return tag_list

    def print_training_tag_word_dict(self):
        print(self.training_set_tag_word)

    def print_training_word_tag_dict(self):
        print(self.training_set_word_tag)

    def print_tag_tag_counts_dict(self):
        print(self.tag_tag_counts_dict)
        print(len(self.tag_tag_counts_dict))

    def print_tags(self):
        print(self.tags)
        print(len(self.tags))

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



    print(bc.viterbi2("the dog"))



    #print("Known err: %s\nUnknown err: %s\nTotal err: %s"%bc.calculate_errors())


main()

