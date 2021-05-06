#-*- coding:utf-8-*-
# USAGE
# Start the server:
#       python run_keras_server.py
# Submit a request via cURL:
#       curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#       python simple_request.py

# import the necessary package                                                                                                   lSampl
import pandas as pd
import numpy as np
import flask
import io
from flask import Flask,request,Response,render_template

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
from classifier import BERTDataset
from classifier import BERTClassifier

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

model = None
app=Flask(__name__)
bertmodel, vocab = get_pytorch_kobert_model()

# 디바이스 설정
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)
@app.route("/",methods=['GET','POST'])
def hello():

        global model
        model = BERTClassifier(bertmodel, dr_rate=1.0)
        checkpoint=torch.load('model_save.pt')
        model.load_state_dict(checkpoint)
        model.to(device)
        
        return render_template('input.html')

@app.route("/post",methods=['GET','POST'])
def test_sentences():
        model.eval()
        sentences = request.form['input']
        #print(sentences)
        output =convert_input_data(sentences)
        print("inputs:",inputs)

        logits=output
        logits=logits.detach().cpu().numpy()
        emotion=np.argmax(logits)
        print("logits",logits)
        print("emotion",emotion)
        return str(emotion)

def convert_input_data(sentences):
                test_data = [sentences]
                tokenizer = get_tokenizer()
                tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

                max_len = 64
                
                test_data = BERTDataset(test_data, 0, tok, max_len, True, False)
                dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=1)

                for token_ids, valid_length, segment_ids in dataloader:
                        token_ids = token_ids.long().to(device)
                        segment_ids = segment_ids.long().to(device)
                        valid_length= valid_length
                        result = model(token_ids, valid_length, segment_ids)

               # return "ge"
                return result

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform(dataset)]

    def __getitem__(self, i):
        return self.sentences[i]

    def __len__(self):
        return (len(self.sentences))


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=6,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
        print(("* Loading Keras model and Flask starting server..."
                "please wait until server has fully started"))
        app.run(host='223.194.46.100', port=5000)
