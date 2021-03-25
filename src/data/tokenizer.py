import string
from config import config

class Tokenizer():
    def __init__(self, filters=None):
        self.filters = filters if filters else string.printable.translate(
            str.maketrans("", "", config.charset)
            )
        self.charset = None
        self.vocab_size = None
        self.char_index = {}
        self.index_char = {}
        self.UNK_TK = "¤"
        self.PAD_TK = "¶"
        self.UNK = self.PAD = None

        self.fit_on_texts(config.charset)

    def fit_on_texts(self, text):
        text = self.PAD_TK + self.UNK_TK + "".join(sorted(set(text)))
        self.charset = text.translate(str.maketrans("", "", self.filters))

        for i in range(len(self.charset)):
            self.char_index[self.charset[i]] = i
            self.index_char[i] = self.charset[i]

        self.UNK = self.char_index[self.UNK_TK]
        self.PAD = self.char_index[self.PAD_TK]
        self.vocab_size = len(self.charset)

    def texts_to_sequences(self, texts):
        result = []
        if isinstance(texts, list):
            for text in texts:
                if not isinstance(text, str):
                    raise TypeError("Only list of strings are valid")
                tempres = []
                for ch in text:
                    tempres.append(char_index[ch] if ch in self.charset else self.UNK)
                result.append(tempres)

        elif isinstance(texts, str):
            tempres = []
            for ch in texts:
                tempres.append(self.char_index[ch] if ch in self.charset else self.UNK)
            result.append(tempres)

        else:
            raise TypeError("Only str or list of strings allowed")

        return result

    def sequences_to_texts(self, batch):  #add top paths     #[batch_size, top_path] = 1 label
        result = []
        for image in batch:
            lines = []
            for top_path in image:
                line = ""
                for index in top_path:
                    line += self.index_char[index] if index in self.index_char.keys() else self.UNK_TK #Watch out
                lines.append(line)
            result.append(lines)
        return result

