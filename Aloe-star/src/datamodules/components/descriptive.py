import torch
from torch.utils.data import Dataset
import transformers

import numpy as np
import json
import os

class descriptive_dataset(Dataset):
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

        with open(qa_file, "r") as h:
            self.qa_data = json.load(h)['data']

        self.extra_check = []
        for i in os.listdir(monet_dir):
            self.extra_check.append(int(i.split(".")[0].split("_")[-1]))

        self.metaqa = {}
        for i in range(len(self.qa_data)):
            if (
                int(self.qa_data[i]["example_file"].split(".")[0].split("_")[-1])
                not in self.extra_check
            ):
                continue
            
            if int(self.qa_data[i]["example_file"].split(".")[0].split("_")[-1]) >= self.filter[0] and int(self.qa_data[i]["example_file"].split(".")[0].split("_")[-1]) < self.filter[1]:
                pass
            else:
                continue
  

            if self.qa_data[i]["example_file"].split(".")[0].split("_")[-1] not in self.metaqa:
                self.metaqa[self.qa_data[i]["example_file"].split(".")[0].split("_")[-1]] = []

            self.metaqa[self.qa_data[i]["example_file"].split(".")[0].split("_")[-1]].append(
                (self.qa_data[i]["question"], self.qa_data[i]["answer"])
            )

        self.data = {}
        en = 0
        for k, v in self.metaqa.items():
            for q, a in v:
                self.data[en] = (q, a, k)
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

        question, answer, idx = self.data[index]
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

