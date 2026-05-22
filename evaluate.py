import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Download required NLTK data
nltk.download("wordnet")
nltk.download("omw-1.4")

# -----------------------------
# Sample test data
# Replace these with your real references and bot outputs
# -----------------------------
references = [
    "You may have a common cold. Rest and drink fluids.",
    "This could be a mild fever. Monitor temperature and stay hydrated.",
    "You might be experiencing a headache due to stress."
]

predictions = [
    "It seems like a common cold. Take rest and drink fluids.",
    "This looks like a mild fever. Stay hydrated and monitor your temperature.",
    "It could be a stress related headache."
]

# -----------------------------
# METEOR Score
# -----------------------------
meteor_scores = []

for ref, pred in zip(references, predictions):
    ref_tokens = ref.split()
    pred_tokens = pred.split()
    score = meteor_score([ref_tokens], pred_tokens)
    meteor_scores.append(score)

avg_meteor = sum(meteor_scores) / len(meteor_scores)

# -----------------------------
# ROUGE Score
# -----------------------------
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

rouge1_list = []
rouge2_list = []
rougeL_list = []

for ref, pred in zip(references, predictions):
    scores = scorer.score(ref, pred)
    rouge1_list.append(scores["rouge1"].fmeasure)
    rouge2_list.append(scores["rouge2"].fmeasure)
    rougeL_list.append(scores["rougeL"].fmeasure)

avg_rouge1 = sum(rouge1_list) / len(rouge1_list)
avg_rouge2 = sum(rouge2_list) / len(rouge2_list)
avg_rougeL = sum(rougeL_list) / len(rougeL_list)

# -----------------------------
# Print individual results
# -----------------------------
print("\n===== Individual Results =====")
for i, (ref, pred, m) in enumerate(zip(references, predictions, meteor_scores), start=1):
    scores = scorer.score(ref, pred)
    print(f"\nTest Case {i}")
    print(f"Reference : {ref}")
    print(f"Prediction: {pred}")
    print(f"METEOR    : {m:.4f}")
    print(f"ROUGE-1   : {scores['rouge1'].fmeasure:.4f}")
    print(f"ROUGE-2   : {scores['rouge2'].fmeasure:.4f}")
    print(f"ROUGE-L   : {scores['rougeL'].fmeasure:.4f}")

# -----------------------------
# Print average results
# -----------------------------
print("\n===== Average Evaluation Results =====")
print(f"Average METEOR Score : {avg_meteor:.4f}")
print(f"Average ROUGE-1 Score: {avg_rouge1:.4f}")
print(f"Average ROUGE-2 Score: {avg_rouge2:.4f}")
print(f"Average ROUGE-L Score: {avg_rougeL:.4f}")