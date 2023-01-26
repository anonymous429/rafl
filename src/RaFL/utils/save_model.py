import os

import torch


def save_checkpoint(state, name, checkpoint_dir):
    filename = '{}.pth.tar'.format(name)
    filepath = os.path.join(checkpoint_dir, filename)
    print('=> Saving checkpoint to {}'.format(filepath))
    torch.save(state, filepath)

