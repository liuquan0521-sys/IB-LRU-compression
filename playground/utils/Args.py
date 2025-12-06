import argparse

from datetime import datetime
def train_options():
    parser = argparse.ArgumentParser(description="Training script.")

    # dataset parameters
    parser.add_argument("--data_file_dir", type=str, default='./Datasets/Hybrids', help="Decoder block configuration")
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--aux_learning_rate", type=float, default=1e-3, help="Auxiliary learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size for testing")
    
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loading workers")
    parser.add_argument("--lmbda", type=float, default=0.0035, help="Lambda value for rate-distortion tradeoff")
    parser.add_argument("--beta", type=float, default=1e-4, help="beta value for KL tradeoff")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size for input images")
    parser.add_argument("--detype", type=str, default=['derainL', 'derainH', 'dehaze', 'denoise15', 'denoise25', 'denoise50'],)
    parser.add_argument("--model_type", type=str, default='base', help="[base, prompt]")
    parser.add_argument("--cuda", type=bool, default=True, help="Use CUDA for training")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")

    # Saving and logging
    parser.add_argument("--base_dir", type=str, default="../experiment_git", help="save root dir")
    parser.add_argument("--experiment", type=str, default="daze_0018", help="experiment dir")
    parser.add_argument("--save", type=bool, default=True, help="Save the model after training")
    parser.add_argument("--clip_max_norm", type=float, default=1.0, help="Maximum norm for gradient clipping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_base", type=str, default=None, help="Path to the checkpoint file")
    parser.add_argument("--checkpoint_prompt", type=str, default=None, help="Path to the checkpoint file")

    # Prompt-related parameters
    parser.add_argument("--INITIATION", type=str, default='random', help="Initiation method for prompts")
    parser.add_argument("--PROJECT", type=int, default=-1, help="Project parameter")
    parser.add_argument("--DROPOUT", type=float, default=0.0, help="Dropout rate for prompts")
    parser.add_argument("--TRANSFER_TYPE", type=str, default='prompt', help="Transfer type")
    parser.add_argument("--WINDOW", type=str, default='same', help="Window type for prompt")

    # Additional parameters
    
    parser.add_argument("--HYPERPRIOR", type=bool, default=False, help="Use hyperprior or not")
    parser.add_argument("--RETURN_ATTENTION", type=bool, default=False, help="Return attention maps or not")
    parser.add_argument("--MASK_DOWNSAMPLE", type=int, default=2, help="Mask downsample factor")
    ###[0] means no prompt in any trans_block
    parser.add_argument("--DECODER_BLOCK", type=int, nargs='+', default=[], help="Decoder block configuration")
    parser.add_argument("--ENCODER_BLOCK", type=int, nargs='+', default=[1,2,3,4], help="Encoder block configuration")

    
    
    
    parser.add_argument(
        "-T",
        "--TEST",
        action='store_true',
        help='Testing'
    )

    args = parser.parse_args()
   
    return args


def test_options():
    parser = argparse.ArgumentParser(description="Testing script.")
    parser.add_argument(
       
        "--save_dir",
        default="elic_test",
        type=str,
        required=False,
        help="Experiment name"
    )
    parser.add_argument(
        "-exp",
        "--experiment",
        default="elic_test",
        type=str,
        required=False,
        help="Experiment name"
    )
    parser.add_argument(
        "--codestream_path",
        default="experiments/elic_0800/codestream/100",
        type=str,
        required=False,
        help="Path to the codestream"
    )
    parser.add_argument(
        "-d",
        "--test_data",
        default="/data/liuquan/DATA1/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
        type=str,
        required=False,
        help="Training dataset"
    )
    parser.add_argument(
        
        "--data_root",
        default="/data/liuquan/DATA1/VOCdevkit/VOC2012/segment",
        type=str,
        required=False,
        help="Training dataset"
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="mse",
        help="Optimized for (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID"
    )
    parser.add_argument(
        "--cuda",
        default=True,
        help="Use cuda"
    )
    parser.add_argument(
        "--save",
        default=True,
        help="Save model to disk"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        type=str,
        help="pretrained model path"
    )
    args = parser.parse_args()
    return args
