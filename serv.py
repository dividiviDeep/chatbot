import socket
import os
import json
import numpy as np
import torch
from dialogLM.model.kogpt2 import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer


arr=[] # 답변전달을 위한list



host = "223.194.46.208" # Server IP
port = 5010 # Port Number

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP 소켓 생성
soc.bind((host, port)) # host와 port로 소켓 연결
soc.listen(5) # 클라이언트 최대 5개 까지 받으며 접속할 때 까지 대기

conn, addr = soc.accept() # 연결 허용, conn 변수에는 클라이언트 소켓이 저장되고, addr에는 클>라이언트 아이피 주소가 저장된다.
print (addr) # 클라이언트 IP 주소 출력
root_path='..'
data_path = f"{root_path}/data/wellness_dialog_for_autoregressive_train.txt"
checkpoint_path =f"{root_path}/checkpoint"
    # save_ckpt_path = f"{checkpoint_path}/kogpt2-wellnesee-auto-regressive.pth"
save_ckpt_path = f"/home/dividivi/flask_dir/dialogLM/checkpoint/kogpt2-catbot-wellness0.pth"
ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)
    # 저장한 Checkpoint 불러오기
checkpoint = torch.load(save_ckpt_path, map_location=device)
model = DialogKoGPT2()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
tokenizer = get_kogpt2_tokenizer()

count = 0
output_size = 200 # 출력하고자 하는 토큰 갯수

while True:
    sent=conn.recv(1024).decode(encoding='utf-8')
    #if sent.isalpha():
    #    print("한국말로 해주세용^^")
    #    print(100 * '-')
    #    continue
    tokenized_indexs = tokenizer.encode(sent)
    input_ids = torch.tensor([tokenizer.bos_token_id,]  + tokenized_indexs +[tokenizer.eos_token_id]).unsqueeze(0)
      # set top_k to 50
    sample_output = model.generate(input_ids=input_ids)

    str= tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs)+1:],skip_special_tokens=True)
    answer=str.split('.')[0]
    print(answer)
    conn.send(answer.encode()) # 클라이언트에 문자열 전송
conn.close() # 소켓 종료
