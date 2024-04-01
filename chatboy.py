import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
from tqdm import tqdm
import fmmap as mmap
import random
import pickle
import transformers
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence

device =  'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)

# Load the BERT uncased tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Get the vocabulary size
vocab_size = len(tokenizer)

block_size = 32     # 
batch_size = 16     # How many batches until model parameters are updated
learning_rate= 3e-3 # Learning rate of the model
num_epochs = 10     # Total amount of iterations for training to loop through
eval_iters = 100    # Amount of iters until loss is reported in training loop
n_embd = 32         # Dimensionality of the input
n_head = 1          # 
dropout = 0.2       # Dropout rate during training
n_layer= 1          # How many layers of blocks
run_time= 60*60*3   # Runtime for training (hours)

class FeedForward(nn.Module):
    """ simple linear layer followed by activation """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout) # Reduce overfit
        )
    
    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Mask

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, H)
        q = self.query(x) # (B, T, H)
        v = self.value(x) # (B, T, H)

        # Compute attention scores ("affinities") [Attention from each token to every other token]
        wei = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5)  # (B, T, T)

        # Create a lower-triangular matrix of size (T, T) as mask
        mask = torch.tril(torch.ones((T, T), device=x.device)).view(1, T, T)

        # Apply the mask to the attention scores
        wei = wei.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get the attention probabilities
        wei = F.softmax(wei, dim=-1)
        
        # Apply dropout to attention weights
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        out = torch.matmul(wei, v)  # (B, T, head_size)
        
        return out
    

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parrallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.heads = nn.ModuleList([Head(head_size) for _ in range (num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) # (B, T, F) --> (B, T, N*H)
        out = self.dropout(self.proj(out)) # (B, T, N*H) --> (B, T, C)
        return out
    

class Block(nn.Module):
    """ Transformer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd //n_head 
        self.sa = MultiHeadAttention(n_head,head_size) # Self-attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        
    def forward(self, x):
        y = self.sa(x)    # (B, T, C); Passing through the multi-head attention
        x = self.ln1(x+y) # (B, T, C); Wrap around
        y = self.ffwd(x)  # (B, T, C) --> (B, T, 4*C) --> (B, T, C); Passing through Feed Forward for non-linearity
        x = self.ln2(x+y) # (B, T, C); Wrap around again
        
        return x



class GPTLanguageModel(nn.Module):
     def __init__ (self, vocab_size):
          super().__init__()
          self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # Token embedding
          self.positional_embedding = nn.Embedding(block_size, n_embd) # Positional Embedding
          self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # Creating our decoder (Block) layers
          self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
          self.lm_head = nn.Linear(n_embd, vocab_size) # Final Linear Layer

          self.apply(self._init_weights)

     def _init_weights(self, module):
          if isinstance(module, nn.Linear):
               torch.nn.init.normal_(module.weight, mean = 0.0, std=0.02)
               if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
          elif isinstance(module, nn.Embedding):
               torch.nn.init.normal_(module.weight, mean =0.0, std=0.02)

     def forward(self, input_ids, labels = None):
         # Getting the batch size (B) and seq_length (T) from the input_ids
          B, T = input_ids.shape

          #Token embeddings
          token_embeddings = self.token_embedding_table(input_ids)              # (B, T, C)

          # Positional Embedding
          position_ids = torch.arange(0,T, dtype=torch.long, device = device)
          positional_embeddings = self.positional_embedding(position_ids)       # (T, C)

          # Combine token and position embeddings
          hidden_states = token_embeddings + positional_embeddings.unsqueeze(0) # (B, T, C)

          # Pass through transformer blocks
          for block in self.blocks:
               hidden_states = block(hidden_states)                             # (B, T, C)

          #Final layer normalization
          hidden_states= self.ln_f(hidden_states)                               #(B, T, C)

          # Obtaining the logits (predictions for each token in the vocabulary)
          logits = self.lm_head(hidden_states)                                  #(B, T, vocab_size)

          # Obtaining Loss
          if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.lm_head.out_features), labels.view(-1))
            return logits, loss
          else:
            return logits, None
     def generate(self, index, max_new_tokens):
        generated = index
        for _ in range(max_new_tokens):
            logits, _ = self.forward(generated)  # No labels provided
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token_ids = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token_ids), dim=1)
        return generated

# Instantiate the model
model = GPTLanguageModel(vocab_size=vocab_size)
optimizer = torch.optim.AdamW(params = model.parameters(), lr = learning_rate) # Utilising Adam's AdamW variant to obtain weight decay/grad independence

# Load the state dictionary
model.load_state_dict(torch.load('gpt_language_model.pth'))

# Make sure to call .to(device) to move the model to the appropriate device, especially if you're using a GPU for inference
model = model.to(device)

# Model use
def input_type():
    answer = ""
    while answer == "":
        answer = input("What would you like to do? (chatbot or qa?)")
        if answer == "chatbot":
            prompt = input("What is your prompt? ")
        elif answer == "qa":
            question = input("What your question? ")
            context = "I'm a knowledgeable assistant, feel free to ask me anything."
            prompt = f"{context}\nQuestion: {question}\nAnswer:"   
        else:
            print("Error: incorrect input, please choose either chatbot or qa.")
            answer = ""
        
    return prompt

prompt = input_type()
# Tokenize the prompt
encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
encoded_prompt = encoded_prompt.to(device)

max_new_tokens = 50  # Adjust as needed
generated_tokens = model.generate(encoded_prompt, max_new_tokens=max_new_tokens)

generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(generated_text)

