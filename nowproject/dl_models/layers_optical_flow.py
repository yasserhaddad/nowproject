import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, raft_small
from nowproject.utils.utils_models import (
    transform_data_for_raft,
    inverse_transform_data_for_raft,
    flow_warp,
    check_data_for_raft
)


class RAFTOpticalFlow(nn.Module):
    def __init__(self, input_time, small_model=False, finetune=False) -> None:
        super().__init__()

        self.raft = raft_small if small_model else raft_large
        self.raft = self.raft(pretrained=True, progress=False)
        if not finetune:
            self.raft = self.raft.eval()
        self.input_time = input_time

    def compute_flow_field(self, x1, x2):
        batch_1 = transform_data_for_raft(x1)
        batch_1, pad = check_data_for_raft(batch_1)
        batch_1 = torch.stack([batch_1, batch_1, batch_1], dim=1)\
                        .reshape(batch_1.shape[0], 3, batch_1.shape[-2], batch_1.shape[-1])
        
        batch_2 = transform_data_for_raft(x2)
        batch_2, pad = check_data_for_raft(batch_2)
        batch_2 = torch.stack([batch_2, batch_2, batch_2], dim=1)\
                        .reshape(batch_2.shape[0], 3, batch_2.shape[-2], batch_2.shape[-1])

        predicted_flows = self.raft(batch_1, batch_2)
        predicted_flows = predicted_flows[-1]
        
        return predicted_flows, pad, batch_1, batch_2


    def forward(self, x):
        flow_fields = []
        batches = []
        out = []
        for i in range(self.input_time - 1):
            predicted_flows, pad, batch_1, batch_2 = self.compute_flow_field(x[:, :, i, :], x[:, :, i+1, :])
            flow_fields.append(predicted_flows)
            batches.append(batch_1)
            if i == self.input_time - 2:
                batches.append(batch_2)
                flow_fields.append(predicted_flows)
            
            del predicted_flows, batch_1, batch_2
        
        for i in range(self.input_time):
            flows_to_apply = flow_fields[i:] if i < len(flow_fields) else [flow_fields[-1]]
            batch = batches[i]
            for flow in flows_to_apply:
                batch = flow_warp(batch, flow.permute(0, 2, 3, 1),
                                  interpolation="nearest", 
                                  padding_mode="reflection")
            batch = inverse_transform_data_for_raft(batch)
            if len(pad) > 0:
                if pad[2] == 0 and pad[3] == 0:
                    batch = batch[:, 0, :, pad[0]:-pad[1]]
                elif pad[1] == 0 and pad[0] == 0:
                    batch = batch[:, 0, pad[2]:-pad[3], :]
                else:
                    batch = batch[:, 0, pad[2]:-pad[3], pad[0]:-pad[1]]
            else:
                batch = batch[:, 0, :, :]
            out.append(batch.unsqueeze(dim=1))
        
        return torch.stack(out, dim=2), flow_fields, batches
        
