import torch
from torch.utils.data import Dataset
import transformers

import numpy as np
import json
import pickle5 as pickle
import os

import logging

class counterfactual_dataset(Dataset):
    def __init__(
        self,
        config,
        answer_map=None,
        monet_dir="",
        annot_dir="",
        qa_file="",
        filter=[0,4000],
        train=True,
    ):

        assert answer_map != None

        self.annot_dir = annot_dir
        self.monet_dir = monet_dir
        self.qa_file = qa_file
        self.config = config
        self.filter = filter
        self.train = train
        
        self.answer_map = answer_map
        self.metaqa = {}
        self.loading_issues = 0

        with open(qa_file, "r") as h:
            self.qa_data = json.load(h)

        self.extra_check = []
        for i in os.listdir(monet_dir):
            self.extra_check.append(int(i.split(".")[0].split("_")[-1]))

        self.get_counterfactual_data(self.qa_data["remove"], "counterfactual-remove")
        self.get_counterfactual_data(self.qa_data["replace"], "counterfactual-replace")
        self.get_counterfactual_data(self.qa_data["add"], "counterfactual-add")

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
    
    def get_counterfactual_data(self, qa_data, qa_type):

        for i in range(len(qa_data)):
            if (
                int(qa_data[i]["example_file"].split(".")[0].split("_")[-1])
                not in self.extra_check
            ):
                self.loading_issues += 1
                continue
            if int(qa_data[i]["example_file"].split(".")[0].split("_")[-1]) >= self.filter[0] and int(qa_data[i]["example_file"].split(".")[0].split("_")[-1]) < self.filter[1]:
                pass
            else:
                continue

            if qa_data[i]["example_file"].split(".")[0].split("_")[-1] not in self.metaqa:
                self.metaqa[qa_data[i]["example_file"].split(".")[0].split("_")[-1]] = []


            q_ = qa_data[i]['question']
            for k in range(len(qa_data[i]["choices"])):
                c_ = qa_data[i]["choices"][k]['choice']
                a_ = qa_data[i]["choices"][k]['answer']

                self.metaqa[qa_data[i]["example_file"].split(".")[0].split("_")[-1]].append(
                    (
                        f"{q_} Choice: {c_}",
                        a_,
                        int(qa_data[i]["example_file"].split(".")[0].split("_")[-1]),
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

        question, answer, idx, _ = self.data[index]
        idx = int(idx)

        monet = self.process_visuals(
            np.load(os.path.join(self.monet_dir, f"example_{idx}.npy"))
        )
        visuals = torch.from_numpy(monet).float()
        target = torch.from_numpy(np.array([self.answer_map[answer]])).long()

        tmp = self.process_data(question, self.config["bert"]["max_len"])
        question_tokens = {
            "ids": tmp["ids"],
            "mask": tmp["mask"],
        }

        return visuals, question_tokens, target[0]

    def __len__(self):
        return len(self.data)

