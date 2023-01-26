from RaFL.lib import ArcSampler

if __name__ == '__main__':
    arc_sampler = ArcSampler('ofa_supernet_mbv3_w12')
    net, net_MACs = arc_sampler.sampling_arc(10)
    arc_sampler.update_supernet("ofa_supernet_resnet50")
    net, net_MACs = arc_sampler.sampling_arc(100)
    print(net_MACs)
