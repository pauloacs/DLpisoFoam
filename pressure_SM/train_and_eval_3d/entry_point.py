import argparse
from pressure_SM.train_and_eval_3d.main import main_train
from pressure_SM.train_and_eval_3d.SM_call import call_SM_main

def train_entry_point():
    parser = argparse.ArgumentParser(description='Train your Surrogate Model.')

    # Required arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--gridded_h5_fn', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--first_sim', type=int, required=True,
                        help='First sim')
    parser.add_argument('--last_sim', type=int, required=True,
                        help='Last sim')    
    parser.add_argument('--last_t', type=int, required=True,
                        help='Last temporal frame to consider')
    parser.add_argument('--num_epoch', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate')
    parser.add_argument('--n_samples_per_frame', type=int, required=True,
                        help='Number of samples per frame')
    parser.add_argument('--var_p', type=float, required=True,
                        help='Var p value')
    parser.add_argument('--var_in', type=float, required=True,
                        help='Var in value')
    parser.add_argument('--model_architecture', type=str, required=True,
                        help='Model architecture')
    parser.add_argument('--outarray_flat_fn', type=str, required=True,
                        help='Path to flat dataset')

    # Optional arguments
    parser.add_argument('--first_t', type=int, default=0,
                        help='1st temporal frame to consider')
    parser.add_argument('--new_model', type=str, default='true',
                        help='Whether to create a new model')
    parser.add_argument('--dropout_rate', type=float, default=None,
                        help="Dropout rate. If None the model won't have dropout layers")
    parser.add_argument('--regularization', type=float, default=None,
                        help="L2 Regularization. If None the model won't do regularization")                        
    parser.add_argument('--beta', type=float, default=0.99,
                        help='Beta value')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--standardization_method', type=str, default='std',
                        help='Standardization method')
    parser.add_argument('--block_size', type=int, default=128,
                        help='Block size')
    parser.add_argument('--grid_res', type=float, default=1e-3,
                        help='Grid resolution value')
    parser.add_argument('--ranks', type=int, default=512,
                        help='Max number of PCs')
    parser.add_argument('--chunk_size', type=int, default=50,
                        help='Chunk size.')

    args = parser.parse_args()

    # Convert args to dictionary
    args_dict = vars(args)

    # Print all the arguments
    print("******** Arguments: **********")
    for key, value in args_dict.items():
        print(f"{key}: {value}")
    print("******************************* \n")

    # Pass arguments to main_train function
    main_train(**args_dict)


def eval_entry_point():
    parser = argparse.ArgumentParser(description='Evaluate your Surrogate Model.')

    # Required arguments
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name')
    parser.add_argument('--var_p', type=float, required=True,
                        help='Var p value')
    parser.add_argument('--var_in', type=float, required=True,
                        help='Var in value')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Dataset path')
    parser.add_argument('--first_sim', type=int, required=True,
                        help='First simulation')
    parser.add_argument('--last_sim', type=int, required=True,
                        help='Last simulation')
    parser.add_argument('--first_t', type=int, required=True,
                        help='First time')
    parser.add_argument('--last_t', type=int, required=True,
                        help='Last time')
    
    # Optional arguments
    parser.add_argument('--delta', type=float, default=1e-3,
                        help='Delta value')
    parser.add_argument('--block_size', type=int, default=128,
                        help='Block size')
    parser.add_argument('--overlap_ratio', type=float, default=0.25,
                        help='Overlap ratio')
    parser.add_argument('--max_num_PC', type=int, default=512,
                        help='Max number of PCs')
    parser.add_argument('--standardization_method', type=str, default='std',
                        help='Standardization method')
    parser.add_argument('--plot_intermediate_fields', action='store_true',
                        help='Plot intermediate fields')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save plots')
    parser.add_argument('--show_plots', action='store_true',
                        help='Show plots')
    parser.add_argument('--apply_filter', action='store_true',
                        help='Apply filter')
    parser.add_argument('--create_GIF', action='store_true',
                        help='Create GIF')

    args = parser.parse_args()

    # Convert args to dictionary
    args_dict = vars(args)

    # Print all the arguments
    print("******** Arguments: **********")
    for key, value in args_dict.items():
        print(f"{key}: {value}")
    print("******************************* \n")

    # Pass arguments to call_SM_main function
    call_SM_main(**args_dict)


if __name__ == "__main__":
  train_entry_point()
  eval_entry_point()
