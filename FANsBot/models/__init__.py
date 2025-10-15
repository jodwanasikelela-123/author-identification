import torch
import os
import json
from torch.nn import functional as F

# torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model name
MODEL_NAME = "lstm.pt"
# Model paths
PYTORCH_LSTM_MODEL_PATH = os.path.join(os.getcwd(), f"models/static/{MODEL_NAME}")

# STATIC FILE NAMES
LABELS = "labels_dict.json"
VOCAB = "vocab.json"
# STATIC FILE PATHS
LABELS_PATH = os.path.join(os.getcwd(), f"models/static/{LABELS}")
VOCAB_PATH = os.path.join(os.getcwd(), f"models/static/{VOCAB}")

with open(LABELS_PATH, "r",  encoding="utf-8") as reader:
    labels_dict = json.load(reader)

with open(VOCAB_PATH, "r",  encoding="utf-8") as reader:
    stoi = json.load(reader)

# SPECIAL TOKENS
PAD_TOKEN = "[pad]"
SOS_TOKEN = "[sos]"
UNK_TOKEN = "[unk]"
EOS_TOKEN = "[eos]"


def text_pipeline(x: str):
    values = list()
    tokens = x.lower().split(" ")
    for token in tokens:
        try:
            v = stoi[token]
        except KeyError:
            v = stoi[UNK_TOKEN]
        values.append(v)
    return values


def inference_preprocess_text(text, max_len=300, padding="pre"):
    assert padding == "pre" or padding == "post", (
        "the padding can be either pre or post"
    )
    text_holder = torch.zeros(
        max_len, dtype=torch.int32
    ) 
    processed_text = torch.tensor(text_pipeline(text), dtype=torch.int32)
    pos = min(max_len, len(processed_text))
    if padding == "pre":
        text_holder[:pos] = processed_text[:pos]
    else:
        text_holder[-pos:] = processed_text[-pos:]
    text_list = text_holder.unsqueeze(dim=0)
    return text_list


classes = list(labels_dict.keys())
authors =   [ "HP Lovecraft",  "Edgar Allan Poe", "Mary Shelley" ]
def predict_author(model, sentence, device):
    model.eval()
    with torch.no_grad():
      tensor = inference_preprocess_text(sentence).to(device)
      length = torch.tensor([len(t) for t in tensor])
      probs = F.softmax(model(tensor, length).squeeze(0))
      top = {
          'label': int(probs.argmax().item()),
          'probability': float(probs.max().item()),
          "class": classes[probs.argmax().item()],
          "author": authors[probs.argmax().item()]
      }
      predictions = [
          {
              'label': i,
              'probability': float(prob),
              "class": classes[i],
              "author": authors[i]
          } for i, prob in enumerate(probs)
      ]
      return {'top': top, 'predictions': predictions}