
EPSILON = 1e-5
import torch
import torch.nn.functional as F

def attention_fusion_weight(tensor1, tensor2):
    # avg, max, nuclear
    f_channel = channel_fusion(tensor1, tensor2)
    f_spatial = spatial_fusion(tensor1, tensor2)

    tensor_f = (f_channel + f_spatial) / 2
    return tensor_f


# select channel
def channel_fusion(tensor1, tensor2):
    # global max pooling
    shape = tensor1.size()
    # calculate channel attention
    global_p1 = channel_attention(tensor1)
    global_p2 = channel_attention(tensor2)

    # get weight map
    global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
    global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

    tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensor_f

def spatial_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f

# channel attention
def channel_attention(tensor):
    # global pooling
    shape = tensor.size()
    pooling_function_avg = F.avg_pool2d
    pooling_function_max = F.max_pool2d

    global_p_avg = pooling_function_avg(tensor, kernel_size=shape[2:])
    global_p_max = pooling_function_max(tensor, kernel_size=shape[2:])

    global_p = global_p_avg + global_p_max

    return global_p


# spatial attention
def spatial_attention(tensor, spatial_type='mean'):

    spatial = []

    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial