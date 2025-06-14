##### Use M3 doing Multilingual Multi-Modal Retrieval
import torch
from visual_bge.modeling import Visualized_BGE
import requests


# import os
# os.environ["HTTP_PROXY"] = "socks5h://127.0.0.1:1080"  # 替换为实际代理
# os.environ["HTTPS_PROXY"] = "socks5h://127.0.0.1:1080"


print("Testing Visualized BGE M3...")

# response = requests.get("https://ipinfo.io")
# print(response.status_code)
#
# response = requests.get("https://google.com")
# print(response.status_code)

model = Visualized_BGE(model_name_bge = "BAAI/bge-m3", model_weight="/home/qhy/MML/dnmsr/Visualized_m3.pth")
print('Loading model done!')
model.eval()
with torch.no_grad():
    query_emb = model.encode(image="./imgs/cir_query.png", text="一匹马牵着这辆车")
    candi_emb_1 = model.encode(image="./imgs/cir_candi_1.png")
    candi_emb_2 = model.encode(image="./imgs/cir_candi_2.png")

sim_1 = query_emb @ candi_emb_1.T
sim_2 = query_emb @ candi_emb_2.T
print(sim_1, sim_2) # tensor([[0.7026]]) tensor([[0.8075]])