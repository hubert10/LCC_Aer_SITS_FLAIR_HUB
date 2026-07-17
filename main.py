import argparse
import sys
import torch
from pathlib import Path
from tasks.stages import training_stage, predict_stage
from tasks.module_setup import build_data_module
from data.utils_data.paths import get_datasets
from utils.messaging import start_msg, end_msg, Logger
from utils.config_io import setup_environment, copy_csv_and_config
from utils.config_display import print_recap

# handling warnings
torch.set_float32_matmul_precision("medium")

argParser = argparse.ArgumentParser()
argParser.add_argument("--config", help="Path to the .yaml config file", required=True)


def main():
    """
    Main function to set up the training and prediction process. It reads the config file, sets up the output folder,
    initiates the training and prediction stages, and tracks emissions if enabled.
    """

    args = argParser.parse_args()
    config, out_dir = setup_environment(args)
    sys.stdout = Logger(
        Path(
            config["paths"]["out_folder"],
            config["paths"]["out_model_name"],
            f'flair-compute{config["paths"]["out_model_name"]}.log',
        ).as_posix()
    )

    start_msg()

    # Define datasets
    dict_train, dict_val, dict_test = get_datasets(config)
    print_recap(config, dict_train, dict_val, dict_test)

    # Copy relevant files for tracking
    if config["saving"]["cp_csv_and_conf_to_output"]:
        copy_csv_and_config(config, out_dir, args)

    # Get LightningDataModule
    dm = build_data_module(
        config, dict_train=dict_train, dict_val=dict_val, dict_test=dict_test
    )

    # Initialize variable for weights
    trained_state_dict = None

    # Training
    if config["tasks"]["train"]:
        trained_state_dict = training_stage(config, dm, out_dir)

    # Inference
    if config["tasks"].get("predict") or config["tasks"].get("metrics_only"):
        out_dir_predict = Path(out_dir, "results_" + config["paths"]["out_model_name"])
        out_dir_predict.mkdir(parents=True, exist_ok=True)
        predict_stage(config, dm, out_dir_predict, trained_state_dict)
    else:
        print("[WARNING] Neither prediction nor metrics_only was enabled. Finishing.")

    end_msg()


if __name__ == "__main__":
    main()




# scp -r nhgnkany@transfer.cluster.uni-hannover.de:/bigwork/nhgnkany/Results/MISR_S2_Aer_LCC_X10_MI_MO_Exp_AUX_Revised_Diff_Denoising_Multi_Res/results/checkpoints/misr/srdiff_highresnet_ltae_ckpt/results_0_/ D:\kanyamahanga\Datasets\MISR_S2_Aer_LCC_X10_MI_MO_Exp_AUX_Revised_Diff_Denoising_Multi_Res

# scp -r nhgnkany@transfer.cluster.uni-hannover.de:/bigwork/nhgnkany/Results/MISR_S2_Aer_LCC_X10_MI_MO_Exp_AUX_Revised_Diff_Denoising_Spec_Matching_Multi_Res/results/checkpoints/misr/srdiff_highresnet_ltae_ckpt/results_0_/ D:\kanyamahanga\Datasets\MISR_S2_Aer_LCC_X10_MI_MO_Exp_AUX_Revised_Diff_Denoising_Spec_Matching_Multi_Res

# scp -r D:\kanyamahanga\Datasets\FLAIR\flair_2_aerial_test.zip nhgnkany@transfer.cluster.uni-hannover.de:/bigwork/nhgnkany/FLAIR



# Task: AERIAL_LABEL-COSIA - Global Metrics:
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# mIoU                 62.8663
# Overall Accuracy     77.5393
# F-score              75.8116
# Precision            77.7982
# Recall               74.9177
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Idx    Class                     IoU        F-score    Precision  Recall     w.TASK          w.AERIAL_RGBI   w.DEM_ELEV      w.SPOT_RGBI     w.SENTINEL2_TS
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# 0      building                  84.2617    91.4587    92.3956    90.5406    1               1               1               1               1
# 1      greenhouse                76.7019    86.8150    83.7500    90.1130    1               1               1               1               1
# 2      swimming_pool             56.4684    72.1787    76.9658    67.9522    1               1               1               1               1
# 3      impervious surface        75.5365    86.0636    84.7620    87.4057    1               1               1               1               1
# 4      pervious surface          57.6090    73.1037    74.5075    71.7518    1               1               1               1               1
# 5      bare soil                 62.5934    76.9938    71.3608    83.5923    1               1               1               1               1
# 6      water                     88.7495    94.0394    92.8627    95.2464    1               1               1               1               1
# 7      snow                      52.5422    68.8887    96.0984    53.6874    1               1               1               1               1
# 8      herbaceous vegetation     54.5750    70.6130    69.3385    71.9352    1               1               1               1               1
# 9      agricultural land         56.2792    72.0239    72.0941    71.9538    1               1               1               1               1
# 10     plowed land               33.5259    50.2163    56.4468    45.2245    1               1               1               1               1
# 11     vineyard                  78.5887    88.0108    84.4598    91.8736    1               1               1               1               1
# 12     deciduous                 72.4124    83.9991    84.1330    83.8656    1               1               1               1               1
# 13     coniferous                64.9035    78.7170    78.8729    78.5617    1               1               1               1               1
# 14     brushwood                 28.2480    44.0522    48.9259    40.0615    1               1               1               1               1


# 0-weighted classes for task
# -----------------------------------
# 15     clear cut
# 16     ligneous
# 17     mixed
# 18     undefined


# Predicting DataLoader 0: 100%|██████████| 50700/50700 [1:29:11<00:00,  9.47it/s]

# #######################################################
# ####################  FINISHED  #######################

# Ending: 2026-03-01  18:46