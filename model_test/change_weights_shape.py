import torch
state_dict = torch.load("pointpillar_7728.pth", map_location="cpu")
state_dict['model_state']["vfe.pfn_layers.0.linear.weight"] = state_dict['model_state']["vfe.pfn_layers.0.linear.weight"].reshape(64,10,1,1)
torch.save(state_dict, "pointpillar_7728_new.pth")
state_dict_new = torch.load("pointpillar_7728_new.pth", map_location="cpu")
