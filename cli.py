import os
import json
import numpy as np
import torch
from flask import Flask,request,Response,render_template
import socket
import time

import pandas as pd
import io
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp


arr=[]

app=Flask(__name__)
model=None


bertmodel, vocab = get_pytorch_kobert_model()

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)


@app.route("/",methods=['POST','GET'])
def hello():

    global model
    model = BERTClassifier(bertmodel, dr_rate=1.0)
    checkpoint=torch.load('model_save.pt')
    model.load_state_dict(checkpoint)
    model.to(device)

    return render_template('test.html')


soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = "127.0.0.1" # 서버 아이피로 변경
port = 5010 # 소켓용 서버 

soc.connect( (host, port) ) # 서버측으로 연결한다.
#print (soc.recv(1024)) # 서버측에서 보낸 데이터 1024 버퍼만큼 받는다.



@app.route("/post",methods=['POST','GET'])
def home():
    while True:
        sent = request.form['msg']
        print(sent)
        soc.send(sent.encode()) # 서버측으로 문자열을 보낸다.
        data=soc.recv(1024).decode(encoding='utf-8')
        print('Received: ',repr(data))

        # 감정 결과 숫자로 출력
        model.eval()
        output =convert_input_data(sent)
        #print("inputs:",inputs)

        logits=output
        logits=logits.detach().cpu().numpy()
        emotion=np.argmax(logits)
        print("logits",logits)
        print("emotion",emotion)



        #감정, 사용자, 챗봇 발화 순서로 입력
        arr.append(emotion)
        arr.append(sent)
        arr.append(data)
        return render_template('test.html',msg=arr)
    soc.close() # 연결 종료


def convert_input_data(sentences):
                test_data = [sentences]
                print(test_data)
                tokenizer = get_tokenizer()
                tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

                max_len = 128

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

        self.sentences = [transform([i[sent_idx]]) for i in dataset]

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

    


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000) #웹출력용 ip와 port로 변경
