"""
To execute this script you should do the following:

python main.py --dataset_path train

Taking into account that dir train must be at the same level as main.py

"""

import argparse
from pathlib import Path
import calculate_characteristics
import task2
import histogram_analysis_method



if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--dataset_path", required=False, default="train",
                    help="Dataset path, by default it should be at the same level as main.py")

    args = vars(ap.parse_args())

    file_path = Path(__file__).parent.absolute()
    dataset_path = str(file_path / Path(args["dataset_path"]))
    # executing tasks:
    dataset_grouped, dataset = calculate_characteristics.calculate_characteristics(dataset_path)
    dataset_train, dataset_valid = task2.dataset_split(dataset_grouped, 0.3)
    # The following line is optional, exporting histogram for: 
    # training dataset signs if (dataset, 0)
    # training dataset signs in separate classes if (dataset_grouped, 1)
    histogram_analysis_method.analyse_obj_hist(dataset, 0)
