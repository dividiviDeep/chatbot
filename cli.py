import os
import json
import numpy as np
import torch
from flask import Flask,request,Response,render_template
import socket
import time

arr=[]

app=Flask(__name__)
@app.route("/",methods=['POST','GET'])
def hello():
    return render_template('test.html')


soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = "223.194.46.208" # 서버 아이피
port = 5010 # 서버 포트

soc.connect( (host, port) ) # 서버측으로 연결한다.
#print (soc.recv(1024)) # 서버측에서 보낸 데이터 1024 버퍼만큼 받는다.

#while True:
#    string= input('Input: ')
#    soc.send(string.encode()) # 서버측으로 문자열을 보낸다.
#    data=soc.recv(1024)
#    print('Received: ',repr(data))
#soc.close() # 연결 종료


@app.route("/post",methods=['POST','GET'])
def home():
    while True:
        sent = request.form['msg']
        print(sent)
        arr.append(sent)#list에 입력 넣기
        soc.send(sent.encode()) # 서버측으로 문자열을 보낸다.
        data=soc.recv(1024).decode(encoding='utf-8')
        print('Received: ',repr(data))
        arr.append(data) #list 에 답변 넣기
        return render_template('test.html',msg=arr)
    soc.close() # 연결 종료


if __name__ == '__main__':
    app.run(host='223.194.46.100', port=5000)
