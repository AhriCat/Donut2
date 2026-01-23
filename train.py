# BASIC TRAINING EXAMPLEfrom datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import TatochromicHybridModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1️⃣ Load dataset and limit to 10,000 examples
ds_full = load_dataset("Juliankrg/Thinking_DataSet_300k", split="train[:10000]")
ds = ds_full.select(range(10_000))  # take first 10k

# 2️⃣ Tokenization
if not model:
  tokenizer = tok  # your TernaryTokenizer instance
  vocab_size = 60000
  dim = 1024
  model = TatochromicHybridModel(vocab_size=vocab_size, dim=dim, depth=2)
  
def tokenize_fn(example):
    prompt  = example["prompt"]
    target  = example["completion"]
    inp_ids = tokenizer.encode(prompt,  max_length=512, pad_to_max=True,
                               return_torch=True, device=device)
    tgt_ids = tokenizer.encode(target, max_length=512, pad_to_max=True,
                               return_torch=True, device=device)
    return {"input_ids": inp_ids, "labels": tgt_ids}

ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
ds.set_format(type="torch", columns=["input_ids", "labels"])

# 3️⃣ DataLoader
dataloader = DataLoader(ds, batch_size=8, shuffle=True)

# 4️⃣ Training setup
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# 5️⃣ Training loop
for epoch in range(8):
    for batch in dataloader:
        inp    = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(inp)  # [B, N, V]
        loss   = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=tokenizer.pad_id,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss {loss.item():.4f}")
