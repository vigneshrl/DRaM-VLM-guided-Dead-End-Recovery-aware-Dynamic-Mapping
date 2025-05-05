import json
predictions = "/gammascratch/vigneshr/v2/predictions.json"
ground_truth = "/gammascratch/vigneshr/train_bags/bag4/annotations.json"
with open(ground_truth,'r') as f:
    gt = json.load(f)
with open(predictions,'r') as f:
    pred = json.load(f)

correct = 0
total = 0
for sample_id in gt:
    if sample_id in pred:
        # Example: compare dead end label
        if gt[sample_id]['is_dead_end'] == pred[sample_id]['is_dead_end']:
            correct += 1
        total += 1

print(f"Accuracy: {correct/total:.2%}")