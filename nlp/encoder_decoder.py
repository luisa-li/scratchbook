from translation_dataset import FraEngDataset
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn 
import torch 
from tqdm import tqdm


class Encoder(nn.Module): 
    
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(input_size=embed_size, 
                          hidden_size=num_hiddens, 
                          num_layers=num_layers, 
                          dropout=dropout)

    def forward(self, X):
        embs = self.embedding(X.t().type(torch.int64))
        outputs, state = self.rnn(embs)
        return outputs, state
    
    
class Decoder(nn.Module):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(input_size=embed_size+num_hiddens, 
                          hidden_size=num_hiddens,
                           num_layers=num_layers, 
                           dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        
    def forward(self, X, state):
        embs = self.embedding(X.t().type(torch.int32))
        enc_output, hidden_state = state
        context = enc_output[-1]
        actual_context = context.repeat(embs.shape[0], 1, 1) 
        embs_and_context = torch.cat((embs, actual_context), -1) 
        outputs, hidden_stat_after = self.rnn(embs_and_context, hidden_state) 
        final_outputs = self.dense(outputs).transpose(0, 1)
        return final_outputs, [enc_output, hidden_stat_after]
    
    
class Seq2Seq(nn.Module): 
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 

    def forward(self, src, target):
        enc_outputs, enc_state = self.encoder(src)
        dec_output, _ = self.decoder(target, [enc_outputs, enc_state])
        return dec_output 
    
if __name__ == "__main__":
    
    dataset = FraEngDataset(num_steps=9)
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    
    eng_vocab_size = len(dataset.eng_vocab)
    fra_vocab_size = len(dataset.fra_vocab)
        
    embed_size = 256    
    num_hiddens = 512  
    num_layers = 2   
    dropout = 0.1       
    epochs = 10
    
    encoder = Encoder(eng_vocab_size, embed_size, num_hiddens, num_layers, dropout)
    decoder = Decoder(fra_vocab_size, embed_size, num_hiddens, num_layers, dropout)
    model = Seq2Seq(encoder, decoder)
    
    device = 'cpu'
    model = model.to(device)
        
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.fra_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}', unit='batch'):
            
            source, decoder, source_len, label = batch
            output = model(source, decoder)
            label = label.contiguous().view(-1)
            output = output.contiguous().view(-1, output.size(-1))
            loss = criterion(output, label)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')
        