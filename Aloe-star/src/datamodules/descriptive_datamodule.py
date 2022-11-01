from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from src.datamodules.components.descriptive import descriptive_dataset
import json
import pickle
import os

import logging

class DescriptiveDataModule(LightningDataModule):
    def __init__(
        self,
        config: dict,
        data_dir: str = "data/",
        answer_map_name: str = "answer_map_des.pickle",
        descriptive_filename: str = "descriptive_gt.json",
        monet_feature_path: str = "",
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
        self.check_answer_map()
    
    def check_answer_map(self):
        answer_map_path = os.path.join(self.hparams.data_dir, self.hparams.answer_map_name)
        if os.path.exists(answer_map_path):
            with open(answer_map_path, "rb") as handle:
                self.answer_map = pickle.load(handle) 
        else:
            logging.info(f"Could not found the answer_map at {answer_map_path}.")
            logging.info("Initiating answer_map generation.")
            
            with open(os.path.join(self.hparams.data_dir, self.hparams.descriptive_filename), "r") as h:
                tmp_ = json.load(h)['data']

            self.answer_map = {}
            ct = 0
            for k in range(len(tmp_)):
                if tmp_[k]['answer'] not in self.answer_map:
                    self.answer_map[k] = ct
                    ct += 1

            with open(answer_map_path, "wb") as handle:
                pickle.dump(self.answer_map, handle)
                
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
            self.data_train = descriptive_dataset(
                config=self.hparams.config["huggingface"], 
                answer_map=self.answer_map,  
                monet_dir=os.path.join(self.hparams.data_dir, self.hparams.monet_feature_path),
                annot_dir="",
                qa_file=os.path.join(self.hparams.data_dir, self.hparams.descriptive_filename),
                filter=[0,4000],
                train=True
            )
            self.data_val = descriptive_dataset(
                config=self.hparams.config["huggingface"], 
                answer_map=self.answer_map,  
                monet_dir=os.path.join(self.hparams.data_dir, self.hparams.monet_feature_path),
                annot_dir="",
                qa_file=os.path.join(self.hparams.data_dir, self.hparams.descriptive_filename),
                filter=[4001,4500],
                train=True
            )
            self.data_test = descriptive_dataset(
                config=self.hparams.config["huggingface"], 
                answer_map=self.answer_map,  
                monet_dir=os.path.join(self.hparams.data_dir, self.hparams.monet_feature_path),
                annot_dir="",
                qa_file=os.path.join(self.hparams.data_dir, self.hparams.descriptive_filename),
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
