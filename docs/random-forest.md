# Random Forest Classifier

## What It Is

Imagine asking 100 people to each independently guess whether a Titanic passenger survived, using a slightly different subset of the information available. Then you take the majority answer. That's roughly what a Random Forest does — except the "people" are simple decision rules (called trees) that the computer builds automatically from your training data.

The key insight is that many imperfect guesses, combined together, tend to produce a better answer than any single perfect-looking guess on its own.

## What It Is Good For

- **Predicting a category** — e.g. "survived" or "didn't survive", "spam" or "not spam"
- **Data that has a mix of number columns and text/category columns** — it handles both without much fuss
- **Getting a good result without a lot of tuning** — the defaults work well in most situations
- **Understanding which inputs matter most** — it can tell you, for example, that passenger sex and class were more useful predictors than the number of siblings aboard

## What It Is Not Great For

- **Explaining a single prediction step-by-step** — with 100 trees involved, it's hard to say exactly *why* it gave a particular answer
- **Predicting values outside the range it was trained on** — if all your training fares were under £300, it struggles with a £500 fare it has never seen
- **Very large datasets with tight speed requirements** — it can be slower and use more memory than simpler approaches

## In This Project

`train.py` trains a Random Forest on Titanic passenger data to predict survival. It builds 100 trees, each limited to 5 levels of depth to keep them simple and prevent the model from just memorising the training data instead of learning general patterns.
