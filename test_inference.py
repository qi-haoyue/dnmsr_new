####### Use Visualized BGE doing multi-modal knowledge retrieval
import torch
from visual_bge.modeling import Visualized_BGE

model = Visualized_BGE(model_name_bge = "BAAI/bge-m3", model_weight="/home/qhy/MML/dnmsr/Visualized_m3.pth")
model.eval()
with torch.no_grad():
    query_emb = model.encode(text="毒品")
    candi_emb_1 = model.encode(text="二甲基色胺dmt+壮阳春药+麻黄碱麻黄素+蒙汗药+迷香闻之就睡+相思皮提炼致幻剂傻瓜式全网最简单最便宜毒品配方方法冰毒原料-使用及制作方法指南大全死藤水相思汤买下自动发货制作方法总共十几种简简单单无需化学基础只要你人不傻就能做出来的", image="/home/qhy/MML/DNM_dataset/exchange_market/pic/full/实体发货商品/GHB迷奸水昏迷水/ 645755fd49587a0750a707681218e37931f0c33f.jpg")
    candi_emb_2 = model.encode(text="full/8/√√√√√√√√√猪肉冰毒毒品，12水培海洛因：如何在没有土壤的情况下种植罂粟√√√√√√√√√/ eb5b30a277f31435852e24af327db98d249100ad.jpg", image="/home/qhy/MML/DNM_dataset/exchange_market/pic/full/实体发货商品/LSD国内发货/ 7b8d23a3dd9288565546ad90b6c6443621f8001d.jpg")
    candi_emb_3 = model.encode(text="猪肉冰毒毒品需要操作的制药操作百科大全.")
    candi_emb_4 = model.encode(text="看图", image="/home/qhy/MML/DNM_dataset/exchange_market/pic/full/实体发货商品/LSD国内发货/ 7b8d23a3dd9288565546ad90b6c6443621f8001d.jpg")
    candi_emb_5 = model.encode(text="猪肉")
    candi_emb_6 = model.encode(text="冰毒")
    candi_emb_7 = model.encode(text="专供柬缅菲台企业对公及四件套高质量货源")
    candi_emb_8 = model.encode(text="毒品")


sim_1 = query_emb @ candi_emb_1.T
sim_2 = query_emb @ candi_emb_2.T
sim_3 = query_emb @ candi_emb_3.T
sim_4 = query_emb @ candi_emb_4.T
sim_5 = query_emb @ candi_emb_5.T
sim_6 = query_emb @ candi_emb_6.T
sim_7 = query_emb @ candi_emb_7.T
sim_8 = query_emb @ candi_emb_8.T
print(sim_1, sim_2, sim_3,sim_4)
print(sim_5, sim_6, sim_7,sim_8)
# tensor([[0.6932]]) tensor([[0.4441]]) tensor([[0.6415]])