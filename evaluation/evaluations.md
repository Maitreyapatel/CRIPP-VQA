# Evaluations

This folder contains the basic evaluation scripts for reference. We assume that all ground truth and prediction files are located inside the same folder which can be passed as input to the scripts. These scripts are provided as a reference. Feel free to modify the scripts according to your liking. 


## Descriptive and Counterfactual Evaluations
We use accuracy as evaluation metric for both of these category of questions. Run following commands to get the accuracy of these tasks:
```bash
# for descriptive only eval
python eval_regular_qa.py --data_path /path/to/json/data/folder/ --descriptive

# for counterfactual only eval
python eval_regular_qa.py --data_path /path/to/json/data/folder/ --counterfactual

# for descriptive+counterfactual eval
python eval_regular_qa.py --data_path /path/to/json/data/folder/ --all
```

## Planning Evaluations
Coming soon!! Thanks for waiting!

## Expected input data format
This script expects to have ground truth `.json` files inside the data_folder. Moreover, prediction file should be in similar format as ground_truth file. 
```
# descriptive file: descriptive_pred.json
{
	"question_id": str(predicted_answer),
	.
	.
	.
}

# counterfactual file: counterfactual_pred.json
{
	"remove": {
		"question_id": [choice1_answer, ..., choicek_answer],
		.
		.
		.
	},
	"replace": {
		"question_id": [choice1_answer, ..., choicek_answer],
		.
		.
		.
	},
	"add": {
		"question_id": [choice1_answer, ..., choicek_answer],
		.
		.
		.
	},
}
```

## Issues
For any question, please reach out via GitHub issues. The best recommended way is to reach out to me via email [maitreya.patel@asu.edu](mailto:maitreya.patel@asu.edu).
