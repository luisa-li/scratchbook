import torch
from vocab import Vocab
from torch.utils.data import Dataset, DataLoader

class FraEngDataset(Dataset):
    
    def __init__(self, num_steps=20):
        self.num_steps = num_steps 
        with open("data/fra-eng/fra.txt") as f:
            raw_text = f.read()
        preprocessed = self._preprocess(raw_text)
        self.english, self.french = self._tokenize(preprocessed)
        self.eng_array, self.eng_vocab, self.eng_valid_len = self._build_array(self.english)
        self.fra_array, self.fra_vocab, _ = self._build_array(self.french, is_tgt=True)
        
    def _preprocess(self, text: str) -> str:
        """
        Preprocess the data from the raw text file into a single cleaned string.
        """
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text.lower())]
        return ''.join(out)
    
    def _tokenize(self, text: str) -> tuple[list[list[str]]]:
        """
        Turn the raw text into a tuple[list[list[str]]], where the outer tuple contains the data in (eng, fra) respectively
        The outer lists of list[list[str]] are sentences, while the inner lists are the individual tokens in that sentence (words).
        """
        eng, fra = [], []
        for line in text.split("\n"):
            parts = line.split("\t")
            if len(parts) == 2:
                p1, p2 = parts[0], parts[1]
                eng.append(p1.split(" ") + ["<eos>"])
                fra.append(p2.split(" ") + ["<eos>"])        
        return (eng, fra)
    
    def _pad_or_trim(self, sentence) -> tuple[list[str], int]:
        """
        Either pads or trims a sentence, depending on self.num_steps.
        Returns the padded/trimmed sentence, and the valid length used for masking.
        """
        if len(sentence) > self.num_steps:
            return sentence[:self.num_steps], self.num_steps
        else:
            return sentence + ["<pad>"] * (self.num_steps - len(sentence)), self.num_steps
    
    def _build_array(self, sentences, is_tgt=False) -> tuple[torch.tensor, Vocab, list[int]]:
        """
        Builds the padded and tokenized arrays for sentences.
        Returns the arrays, the vocabulary created, and the valid lengths for loss masking.
        """
        padded_sentences, valid_length = [], []
        for s in sentences:
            sentence, length = self._pad_or_trim(s)
            padded_sentences.append(sentence)
            valid_length.append(length)
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        vocab = Vocab(sentences, min_freq=2)
        array = torch.tensor([vocab[s] for s in padded_sentences])
        return array, vocab, valid_length
        
    def __getitem__(self, index):
        return self.eng_array[index], self.fra_array[index,:-1], self.eng_valid_len[index], self.fra_array[index,1:]
    
    def __len__(self):
        return len(self.eng_valid_len)
    
if __name__ == "__main__":
        
    dataset = FraEngDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)

    src, tgt, src_valid_len, label = next(iter(dataloader))
    print('Source:', src.type(torch.int32))
    print('Decoder input:', tgt.type(torch.int32))
    print('Source length:', src_valid_len.type(torch.int32))
    print('Label:', label.type(torch.int32))