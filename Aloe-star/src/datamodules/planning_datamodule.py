from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from src.datamodules.components.planning import planning_dataset
import pandas as pd
import pickle
import os

import logging

class PlanningDataModule(LightningDataModule):
    def __init__(
        self,
        config: dict,
        data_dir: str = "data/",
        answer_map_planning_name: str = "answer_map_planning.pickle",
        monet_feature_path: str = "",
        counterfactual_path: str = "",
        predefined_objects: str = "",
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.check_answer_map()

    @property
    def num_classes(self) -> int:
        return len(self.answer_map)

    def prepare_data(self):
        """Check whether answer_map is available or not.
        """
        self.qa_files = [
            [
                os.path.join(self.hparams.counterfactual_path, "planning_remove_flt_v6"),
                "planning-remove",
            ],
            [
                os.path.join(self.hparams.counterfactual_path, "planning_replace_flt_v6"),
                "planning-replace",
            ],
            [
                os.path.join(self.hparams.counterfactual_path, "planning_add_flt_v6"),
                "planning-add",
            ],
        ]
        return
    
    def check_answer_map(self):
        answer_map_path = os.path.join(self.hparams.data_dir, self.hparams.answer_map_planning_name)
        if os.path.exists(answer_map_path):
            with open(answer_map_path, "rb") as handle:
                self.answer_map = pickle.load(handle) 
        else:
            logging.info(f"Could not found the answer_map at {answer_map_path}.")
            logging.info("Initiating answer_map generation.")
            
            def get_color_name(tmp):
                types_of_color = {
                    "olive1": {"r": 0.5, "g": 0.5, "b": 0, "a": 1},
                    "purple1": {"r": 0.5, "g": 0, "b": 0.5, "a": 1},
                    "teal1": {"r": 0, "g": 0.5, "b": 0.5, "a": 1},
                    "olive2": "{'r': 0.5, 'g': 0.5, 'b': 0, 'a': 1}",
                    "purple2": "{'r': 0.5, 'g': 0, 'b': 0.5, 'a': 1}",
                    "teal2": "{'r': 0, 'g': 0.5, 'b': 0.5, 'a': 1}",
                }
                return (
                    list(types_of_color.keys())[list(types_of_color.values()).index(tmp)]
                    .replace("1", "")
                    .replace("2", "")
                )

            base_csv = pd.read_csv(os.path.join(self.hparams.data_dir, self.hparams.predefined_objects))
            answer_map_planning = {}
            ct = 0
            for i in range(len(base_csv)):
                answer_map_planning[
                    "{} {} {}".format(
                        get_color_name(base_csv["color"][i]),
                        base_csv["material"][i].split("_")[0],
                        base_csv["name"][i],
                    )
                ] = ct
                ct += 1
            answer_map_planning["None"] = -1

            with open(answer_map_path, "wb") as handle:
                pickle.dump(answer_map_planning, handle)

            self.answer_map = answer_map_planning

                
            try:
                assert len(self.answer_map)==self.hparams.config["max_labels"], "max_labels size mismatch"
            except:
                logging.error(f"You max_label size is mismatch. Try running the same command with max_labels={len(self.answer_map)}.")
                assert len(self.answer_map)==self.hparams.config["max_labels"], "max_labels size mismatch"


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = planning_dataset(
                config=self.hparams.config["huggingface"], 
                answer_map=self.answer_map,  
                monet_dir=self.hparams.monet_feature_path,
                annot_dir="",
                qa_files=self.qa_files,
                filter=[0,4000],
                train=True
            )
            self.data_val = planning_dataset(
                config=self.hparams.config["huggingface"], 
                answer_map=self.answer_map,  
                monet_dir=self.hparams.monet_feature_path,
                annot_dir="",
                qa_files=self.qa_files,
                filter=[4001,4500],
                train=True
            )
            self.data_test = planning_dataset(
                config=self.hparams.config["huggingface"], 
                answer_map=self.answer_map,  
                monet_dir=self.hparams.monet_feature_path,
                annot_dir="",
                qa_files=self.qa_files,
                filter=[4501,5000],
                train=True
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
