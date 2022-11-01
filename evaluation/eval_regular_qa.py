from tqdm import tqdm
import json
import os

import argparse


def add_basic_args(parser):
    parser.add_argument('--data_path', '-dp', type=str.lower, default=None, required=True,
        help='path to the folder containing ground truth and prediction files.')
    parser.add_argument('--descriptive', '-d', action='store_true')
    parser.add_argument('--counterfactual', '-c', action='store_true')
    parser.add_argument('--planning', '-p', action='store_true')
    
    parser.add_argument('--all', '-a', action='store_true')
    
    
    
    
def main(args):
    if args.planning:
        print("This script do not support planning based evaluations. Exiting ....!")
        return
    
    if args.all:
        args.descriptive=True
        args.counterfactual=True
    
    if args.descriptive:
        print("Starting descriptive evaluations!")
        total = 0
        correct = 0
        
        with open(os.path.join(args.data_path, "descriptive_gt.json"), "r") as h:
            gt_data = json.load(h)
        gt_qid2ans = {}
        for d in gt_data['data']:
            if d['question_id'] in gt_qid2ans:
                print(f"We found duplicate entry. Is this a mistake? Ignoring the qid: {d['question_id']}")
            else:
                gt_qid2ans[d['question_id']]=d['answer']
        
        with open(os.path.join(args.data_path, "descriptive_pred.json"), "r") as h:
            pred_data = json.load(h)
            
        pgt = set()
        for k,v in tqdm(pred_data.items()):
            if k in pgt:
                print(f"We found duplicate entry. Is this a mistake? Ignoring the qid: {d['question_id']}")
            else:
                pgt.add(k)
                if str(gt_qid2ans[int(k)])==str(v):
                    correct+=1
                total+=1
        
        print("="*40)
        print(f"Descriptive Acc: {correct*100/total}")
        print("="*40)
        
        
    if args.counterfactual:
        print("Starting counterfactual evaluations!")
        pfms = {
            "remove": {
                "po": {"total":0, "correct":0},
                "pq": {"total":0, "correct":0},
            },
            "replace": {
                "po": {"total":0, "correct":0},
                "pq": {"total":0, "correct":0},
            },
            "add": {
                "po": {"total":0, "correct":0},
                "pq": {"total":0, "correct":0},
            },
        }
        
        with open(os.path.join(args.data_path, "counterfactual_gt.json"), "r") as h:
            gt_data = json.load(h)
            
        with open(os.path.join(args.data_path, "counterfactual_pred.json"), "r") as h:
            pred_data = json.load(h)
            
        def get_action_results(gt, pred):
            gt_qid2ans = {}
            for d in gt:
                if d['question_id'] in gt_qid2ans:
                    print(f"We found duplicate entry. Is this a mistake? Ignoring the qid: {d['question_id']}")
                else:
                    gt_qid2ans[d['question_id']] = [ch['answer'] for ch in d['choices']]
            
            po = {"total":0, "correct":0}
            pq = {"total":0, "correct":0}
            
            pgt = set()
            for k,v in tqdm(pred.items()):
                if k in pgt:
                    print(f"We found duplicate entry. Is this a mistake? Ignoring the qid: {d['question_id']}")
                else:
                    pgt.add(k)
                    if len(gt_qid2ans[int(k)])!=len(v):
                        print(f"We found the mismatch in choices. Ignoring the qid: {d['question_id']}")
                    else:
                        for a,b in zip(gt_qid2ans[int(k)], v):
                            if str(a).lower()==str(b).lower():
                                po['correct']+=1
                            po['total']+=1
                        if po['total']==po['correct']:
                            pq['correct']+=1
                        pq['total']+=1
            return po, pq
        
        p1, p2 = get_action_results(gt_data['remove'], pred_data['remove'])
        pfms["remove"]["po"]=p1
        pfms["remove"]["pq"]=p2
        
        p1, p2 = get_action_results(gt_data['replace'], pred_data['replace'])
        pfms["replace"]["po"]=p1
        pfms["replace"]["pq"]=p2
        
        p1, p2 = get_action_results(gt_data['add'], pred_data['add'])
        pfms["add"]["po"]=p1
        pfms["add"]["pq"]=p2
        
        print("="*40)
        print("Followings are counterfactual results:")
        print("="*40)
        
        for k,v in pfms.items():
            print(f"Action: {k}")
            if pfms[k]["po"]["total"]==0:
                print(f"We received zero ttoal predictions for {k} action.")
                continue
            po = pfms[k]["po"]["correct"]*100/pfms[k]["po"]["total"]
            pq = pfms[k]["pq"]["correct"]*100/pfms[k]["pq"]["total"]
            print(f"Accuracy => PO: {po} \t PQ: {pq}")
    
    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Descriptive & Counterfactual evaluations. Please refer to the readme for file structure.')
    add_basic_args(parser)
    args = parser.parse_args()
    main(args)