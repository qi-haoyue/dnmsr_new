import torch
model_weight_path = "//home/qhy/MML/dnmsr/Visualized_m3.pth"
state_dict = torch.load(model_weight_path, map_location='cpu')
print("Keys in the loaded state_dict:")
for key in state_dict.keys():
    print(key)
# 也可以保存到文件查看
# with open("checkpoint_keys.txt", "w") as f:
#     for key in state_dict.keys():
#         f.write(key + "\n")