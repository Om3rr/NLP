from nltk.corpus import brown
nltk.download('brown')
# constants
PERCENTAGE = 10

class BrownCorpus(object):

    def __init__(self, percentage):
        # extract the last PERCENTAGE from brown corpus to be test set
        # the first 100 - PERCENTAGE from brown corpus to be the training set
        brown_news_tagged = brown.tagged_sents(categories='news')
        brown_news_tagged_size = len(brown_news_tagged)
        training_set_size = round(brown_news_tagged_size * percentage / 100)
        self.training_set = brown_news_tagged[-training_set_size:]
        self.test_set = brown_news_tagged[:brown_news_tagged_size - training_set_size]

        # initialize the dictionary training_set_word_tag such that for each tuple
        # of word and tag as a key, we will have a value which represents the count
        # (= number of occurrences) of the tag following to the word in all sentences.
        # so, for each pair, if it's the first time we encountered initialize the value
        # to be 1, otherwise, value++
        self.training_set_word_tag = {}
        for sentence in self.training_set:
            for word, tag in sentence:
                if (word, tag) in self.training_set_word_tag:
                    self.training_set_word_tag[(word, tag)] += 1
                else:
                    self.training_set_word_tag[(word, tag)] = 1

    def get_list_most_suitable_tag_word(self):
        """
        p(tag|word) = count(word, tag) / count(word)
        :return: list of word with tag which max p(tag | word)
        """
        # initialize words_freq dict such that for each word as a key we will have
        # number of occurrences as a value (= count)
        words_freq = {}
        for sentence in self.training_set:
            for t in sentence:
                words_freq[t] = 1 if t not in words_freq else words_freq[t]+1


        self.word_tag_probabilities_dict = {}
        for t, t_freq in self.training_set_word_tag.items():
            count_word = words_freq[t[0]]
            res = t_freq / count_word
            self.word_tag_probabilities_dict[(t[0], t[1])] = res

        # should think how to do it more efficient!!! it's a naive solution right now
        self.word_tag_max_probability_dict = {}
        for word in words_freq: #iterate over keys
            w, t, v = self.get_max_val_for_word(word)
            self.word_tag_max_probability_dict[(w,t)] = v

        print(self.word_tag_max_probability_dict)

## we can make this function easier to read using filter and max (it will be 1 liner or 2)
    def get_max_val_for_word(self, word):
        max_val = 0
        tag = ""
        for tuple, value in self.word_tag_probabilities_dict.items():
            if tuple[0] == word:
                if value > max_val:
                    max_val = value
                    tag = tuple[1]
        return word, tag, max_val


def main():
    # initialize brown corpus training set and test set, test data will be the last
    # PERCENTAGE
    bc = BrownCorpus(PERCENTAGE)

    # return a list such that for each word we will have the most common tag and the
    # probability of p(tag|word)
    bc.get_list_most_suitable_tag_word()

main()

