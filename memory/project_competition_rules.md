---
name: Kaggle Competition Rules
description: Hard constraints for the 1142 Medical Condition Classification competition — what is and isn't allowed
type: project
originSessionId: 1468f535-4568-49c1-841d-6444faf62560
---
This Kaggle competition has strict rules that must never be violated:

1. **Forbidden**: Searching the internet or any external channel to obtain ground-truth labels for the dataset and using them in submissions.
2. **Forbidden**: Analyzing or extracting features from the test set's ground truth and using them in submissions.
3. Violation = score nullified (zero).

**Why:** The competition uses a public dataset, so the labels could theoretically be found online. The rules explicitly prohibit this.

**How to apply:**
- NEVER suggest looking up or using externally sourced labels for kaggle_testset.csv.
- NEVER suggest any technique that uses test set answers to inform predictions.
- Using kaggle_trainset.csv labels for training/evaluation is fully allowed (it's the official training set).
- All unsupervised, zero-shot, and supervised methods trained solely on kaggle_trainset.csv are allowed.
