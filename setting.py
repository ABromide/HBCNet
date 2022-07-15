import argparse

def args_parser():
    r"""
    This file serves as a global enviornment for setting up a simulation.
    """
    parser = argparse.ArgumentParser()

    ## Training parameters
    parser.add_argument('--server_epochs', type=int, default=100,
                        help="Number of Epochs")    
    parser.add_argument('--server_batch', type=int, default=100,
                        help="Batch Size")
    parser.add_argument('--server_lr', type=int, default=0.001,
                        help="Learning Rate")                                                      
    
    ## Attack Type
    parser.add_argument('--attack', type=str, default="regularized",
                        help="either 'parameterized' or 'regularized'")  
    ## Etc.
    parser.add_argument('--root_dir', type=str, default="hbcnets",
                        help="The root directory for saving data and results.")    
    # parser.add_argument('--gpu', default=None,
    #                     help="When using GPU, set --gpu=0")
    # parser.add_argument('--device', default='cpu',
    #                     help="When using GPU, set --device='cuda'")
    parser.add_argument('--gpu', default=0,
                        help="When using GPU, set --gpu=0")
    parser.add_argument('--device', default='cuda',
                        help="When using GPU, set --device='cuda'")

    args = parser.parse_args()
    return args