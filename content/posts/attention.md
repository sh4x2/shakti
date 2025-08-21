
---
title: Attention
summary: Coding Multi-Head Attention mechanism in Pytorch
date: 2025-08-21
---


![[static/images/Pasted image 20250821144158.png]]



In python dictionaries, we lookup values by a key e.g. `dict[key]`. We only grab one value for the key that matches with a query.

The attention mechanism allows us to lookup values from multiple keys and return their weighted sum.

We can think of Attention as a soft dictionary look up, in which we compare the query $Q$ to each key $K$ and then retrieve the corresponding value $V$


## Scaled Dot-Product Attention
![[static/images/Pasted image 20250821153514.png]]
The basis of attention mechanism is the scaled dot-product attention equation:
$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

The **dot-product** $QK^T$ tells us how close the query vector $Q$ is to the key vector $K$. If the vectors are close together the dot-product is higher.

Now sometimes when you do dot-product of vectors which are larger in size, you will end up getting extremely high values. So, as the size of vectors $d_k$ increase - the values will become higher. Therefore the dot-product is **scaled** by $\sqrt{d_k}$ and then passing it through a $softmax$ function over all the keys.

The final term is then a dot-product with the value vector $V$.

Here is the python code
```python
def scaled_dot_product_attention(query, key, value):
	"""
	Performs the scaled dot-product attention.
	
	Args:
		query: (num_samples, sequence_length, embedding_size)
		key: (num_samples, sequence_length, embedding_size)
		value: (num_samples, sequence_length, embedding_size)
	
	Returns:
		output: (num_samples, sequence_length, embedding_size)
	"""

    # get the dimension of the key vector
    dim_k = key.size(-1)
    
    # dot-product with the keys, keys are transposed
    dot_product = torch.bmm(query, key.transpose(1, 2))
    
    # scaling the dot_product
    attention_weights = dot_product / sqrt(dim_k)
    
    # softmax over the sequence_length
    scores = F.softmax(attention_weights, dim=-1)
    
    # dot-product with the values
    output = torch.bmm(scores, value)
	return output
```


## Multi-Head Attention

![[static/images/Pasted image 20250821153458.png]]


Before moving on to **multi-head attention** let's look at a single head. 

The single **attention head** is same as the scaled dot-product attention with one modification. Instead of doing the scaled dot-product attention only once, this operation is repeated a number of times (think about different channels) and the final output is concatenated. Each of the $Q$, $K$ and $V$ vectors are passed through linear layer.  We define a `AttentionHead` Module in Pytorch below.

```python
class AttentionHead(nn.Module):

    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_states):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_states),
            self.k(hidden_states),
            self.v(hidden_states)
        )
        return attn_outputs
```

With one `AttentionHead` defined we can increase this operation `num_heads` times based on the configuration required.

The output from each `AttentionHead` is concatenated across these channels and then again linearly projected into the output (to get the desired shape of output vector).

```python
class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads

		# create num_heads times AttentionHead
		self.heads = nn.ModuleList(
            [
                AttentionHead(embed_dim, head_dim) for _ in range(num_heads)
            ]
        )
		
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):

		# apply num_head time AttentionHead and then concatencate them
        x = torch.cat([head(hidden_state) for head in self.heads], dim=-1)
        
        # linearly project into the output
        x = self.output_linear(x)
        return x
```


## Putting it all together

We are going to perform a self-attention on a text

```python
text = "time flies like an arrow"
```

First the imports
```python
from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
```

Let's use BERT for tokenisation and getting the embeddings for the tokens.
```python
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config = AutoConfig.from_pretrained(model_ckpt)

# tokenize the input text
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

# get the embeddings
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
input_embs = token_emb(inputs.input_ids)
```

Create the `MultiHeadAttention` layer with the config.
```python
# use model config from the beginning
multihead_attn = MultiHeadAttention(config)

# attention outputs concatenated from 12 heads
attn_output = multihead_attn(input_embs)
```



