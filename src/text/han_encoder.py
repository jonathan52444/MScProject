# src/text/han_encoder.py

from __future__ import annotations
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class HANBrief(nn.Module):
    """
    Hierarchical Attention encoder for CT.gov Brief Summary.
    Word-embeddings: BioBERT (frozen) → BiGRU → word-attention.
    Sentence-embeddings: BiGRU → sent-attention → doc vector.
    Returns a fixed-length (2*H) vector.
    """
    def __init__(self,
                 bert_name: str = "dmis-lab/biobert-base-cased-v1.1",
                 h: int = 128,
                 max_sent: int = 40,
                 max_tok : int = 32):
        super().__init__()
        self.tok   = AutoTokenizer.from_pretrained(bert_name)
        self.bert  = AutoModel.from_pretrained(bert_name)
        for p in self.bert.parameters():  # freeze
            p.requires_grad = False

        self.h = h
        self.max_sent = max_sent
        self.max_tok  = max_tok

        self.word_gru = nn.GRU(input_size=768, hidden_size=h,
                               batch_first=True, bidirectional=True)
        self.word_att = nn.Linear(2*h, 1, bias=False)

        self.sent_gru = nn.GRU(input_size=2*h, hidden_size=h,
                               batch_first=True, bidirectional=True)
        self.sent_att = nn.Linear(2*h, 1, bias=False)

    # Helpers
    def _encode_sentence(self, s: str, device: str):
        ids = self.tok.encode_plus(
            s, add_special_tokens=True, truncation=True,
            max_length=self.max_tok, return_tensors="pt")
        ids = {k: v.to(device) for k,v in ids.items()}
        with torch.no_grad():
            out = self.bert(**ids).last_hidden_state[:,0]  # CLS
        return out.squeeze(0)            # (768,)

    # Forward
    def forward(self, batch_text: list[str]):         # len = B
        device = next(self.parameters()).device
        docs = []
        for txt in batch_text:
            sents = txt.split(". ")[:self.max_sent]   # crude split
            if not sents:
                docs.append(torch.zeros(2*self.h, device=device))
                continue
            # 1) word-level CLS embeddings
            word_vecs = torch.stack(
                [self._encode_sentence(s, device) for s in sents], dim=0
            )                                          # (S, 768)

            # 2) BiGRU + word-attention → sent vecs
            h_word,_ = self.word_gru(word_vecs.unsqueeze(0)) #(1,S,2h)
            α = torch.softmax(self.word_att(h_word).squeeze(-1), dim=1)
            sent_vec = (α.unsqueeze(-1) * h_word).sum(1)      #(1,2h)

            # replicate for all sentences (simple trick)
            sent_stack = sent_vec.repeat(len(sents),1).unsqueeze(0)

            # 3) sentence-BiGRU + att → doc vec
            h_sent,_ = self.sent_gru(sent_stack)              #(1,S,2h)
            β = torch.softmax(self.sent_att(h_sent).squeeze(-1), dim=1)
            doc_vec = (β.unsqueeze(-1) * h_sent).sum(1).squeeze(0) #(2h,)

            docs.append(doc_vec)
        return torch.stack(docs, dim=0)   # (B, 2h)
