import torch
from torch.utils.data import Dataset
import transformers

import numpy as np
import pandas as pd
import pickle5 as pickle
import os

import logging

class planning_dataset(Dataset):
    def __init__(
        self,
        config,
        answer_map=None,
        monet_dir="",
        annot_dir="",
        qa_files=[],
        filter=[0,4000],
        train=True,
    ):

        assert answer_map != None

        self.annot_dir = annot_dir
        self.monet_dir = monet_dir
        self.qa_files = qa_files
        self.config = config
        self.filter = filter
        self.train = train
        
        self.answer_map = answer_map
        self.answer_map_direction = {
            "left": 0,
            "right": 1,
            "front": 2,
            "back": 3,
            "None": -1,
        }
        self.answer_map_action = {
            "Remove": 0,
            "Replace": 1,
            "Add": 2,
        }

        self.metaqa = {}
        self.loading_issues = 0

        self.extra_check = []
        for i in os.listdir(monet_dir):
            self.extra_check.append(int(i.split(".")[0].split("_")[-1]))

        for qa_file, qa_type in self.qa_files:
            self.get_counterfactual_data(qa_file, qa_type)

        logging.warn(f"We had to face the challenge in loading total {self.loading_issues} documents.")

        self.data = {}
        en = 0
        answers = []
        for _, v in self.metaqa.items():
            for q, a, k, t in v:
                self.data[en] = (q, a, k, t)
                answers.append(a)
                en += 1

        self.frames = [
            i
            for i in range(
                0, config["video"]["max_frames"], config["video"]["interval"]
            )
        ]
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", max_len=self.config["bert"]["max_len"], truncation=True
        )
    
    def get_counterfactual_data(self, qa_file, qa_type):
        self.qa_data = os.listdir(qa_file)

        for i in range(len(self.qa_data)):
            if (
                int(self.qa_data[i].split(".")[0].split("_")[-1])
                not in self.extra_check
            ):
                self.loading_issues += 1
                continue
            if int(self.qa_data[i].split(".")[0].split("_")[-1]) >= self.filter[0] and int(self.qa_data[i].split(".")[0].split("_")[-1]) < self.filter[1]:
                pass
            else:
                continue

            if self.qa_data[i].split(".")[0].split("_")[-1] not in self.metaqa:
                self.metaqa[self.qa_data[i].split(".")[0].split("_")[-1]] = []

            try:
                with open(os.path.join(qa_file, self.qa_data[i]), "rb") as handle:
                    tmp_data_ = pickle.load(handle)
            except:
                self.loading_issues += 1
                continue

            for j in range(len(tmp_data_["questions"][0])):
                q_ = tmp_data_["questions"][0][j]
                a_ = tmp_data_["answers"][j]

                self.metaqa[self.qa_data[i].split(".")[0].split("_")[-1]].append(
                    (
                        q_,
                        a_,
                        int(self.qa_data[i].split(".")[0].split("_")[-1]),
                        qa_type,
                    )
                )

        return

    def process_data(self, text, max_len):
        text = str(text)

        inputs = self.tokenizer.encode_plus(
            text, None, add_special_tokens=True, max_length=max_len, truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        padding_length = max_len - len(ids)
        assert padding_length >= 0
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
        }

    def process_visuals(self, vis):
        assert vis.shape[0] == 25
        assert vis.shape[1] == 8
        return vis.reshape((-1, 512))

    def __getitem__(self, index):

        question, answer, idx, tp = self.data[index]
        o1, o2, d, action = answer.split("##")

        monet = self.process_visuals(
            np.load(os.path.join(self.monet_dir, f"example_{idx}.npy"))
        )
        visuals = torch.from_numpy(monet).float()

        target_o1 = torch.from_numpy(np.array([self.answer_map[o1]])).long()
        target_o2 = torch.from_numpy(np.array([self.answer_map[o2]])).long()
        target_d = torch.from_numpy(np.array([self.answer_map_direction[d]])).long()
        target_action = torch.from_numpy(
            np.array([self.answer_map_action[action]])
        ).long()

        tmp = self.process_data(question, self.config["bert"]["max_len"])
        question_tokens = {
            "ids": tmp["ids"],
            "mask": tmp["mask"],
        }

        return (
            visuals,
            question_tokens,
            (target_o1[0], target_o2[0], target_d[0], target_action[0]),
        )

    def __len__(self):
        return len(self.data)

