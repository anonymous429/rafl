import os

import torch
# from torchvision.models import vgg16, vgg11
from torchvision import models

from . import resnet
from . import SimpleCNN, SimpleCNNMNIST
from . import vgg11, vgg16

# def init_fl(n_parties,model_name, args):
#     if args.env == 'multi-model':
#         nets = {net_i: None for net_i in range(n_parties)}
#         net_name = ['resnet20','resnet32','resnet44']
#         for net_i in range(n_parties):
#             import random
#             net_type = net_name[random.randint(0, 2)]
#             net = resnet.__dict__[net_type]()
#             if args.ckpt_path is not None:
#                 # path = os.path.join(data_root, "pretrained_models",'resnet56-4bfd9763.th')
#                 checkpoint = torch.load(args.ckpt_path, map_location=args.device)
#                 sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
#                 net.load_state_dict(sd)

#             net = torch.nn.DataParallel(net)
#             nets[net_i] = net

#         return nets, None, None
#     else:
#         return init_nets(n_parties,model_name, args)


# def multi_model_res_family(n_parties):
def init_nets(n_parties,model_name, args):

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):

        if model_name == "vgg":
            net = vgg11()

        # if model_name == "resnet56":
        #     net = resnet.__dict__['resnet56']()


        elif model_name == "resnet44":
            net = resnet.__dict__['resnet44']()


        elif model_name == "resnet110":
            net = resnet.__dict__['resnet110']()


        elif model_name == "resnet32":
            if args.dataset=='cifar10':
                net = resnet.__dict__['resnet32']()
            elif args.dataset=='cifar100':
                net = resnet.__dict__['resnet32_cifar100']()

        elif model_name == "resnet20":
            # net = resnet.__dict__['resnet20']()
            if args.dataset=='cifar10':
                net = resnet.__dict__['resnet20']()
            elif args.dataset=='cifar100':
                net = resnet.__dict__['resnet20_cifar100']()


        elif args.model == "vgg16":
            net = vgg16()

        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)

        elif args.model =='????':
            raise NotImplementedError

        else:
            raise NotImplementedError

        if args.ckpt_path is not None:
            # path = os.path.join(data_root, "pretrained_models",'resnet56-4bfd9763.th')
            checkpoint = torch.load(args.ckpt_path,map_location=args.device)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)

        net = torch.nn.DataParallel(net)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


if __name__ == '__main__':
    import torch.nn.functional as F
    net = resnet.__dict__['resnet20']()
    batch_size  = 1
    inputs = torch.rand(batch_size,3,64,64)
    output = net(inputs)
    print(output.shape)
    lable = torch.zeros(batch_size).long()
    # print(lable)
    b = F.cross_entropy(output, lable)

    print("output shape:", output.shape, "lable shape:",lable.shape)