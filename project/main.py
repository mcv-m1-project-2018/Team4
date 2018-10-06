"""
To execute this script you should do the following:

python main.py --dataset_path train

Taking into account that dir train must be at the same level as main.py

"""

import argparse
from pathlib import Path
import task1
import task2


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-dp", "--dataset_path", required=False, default="train",
                    help="Dataset path, by default it should be at the same level as task1.py")

    args = vars(ap.parse_args())

    file_path = Path(__file__).parent.absolute()
    dataset_path = str(file_path / Path(args["dataset_path"]))
    # executing tasks:
    dataset_grouped = task1.calculate_characteristics(dataset_path)
    dataset_train, dataset_valid = task2.dataset_split(dataset_grouped, 0.3)
