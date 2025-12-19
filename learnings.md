A basic ml model pipeline
Load data
↓
Modify / clean data
↓
Split data
↓
Fit model
↓
Predict
↓
Evaluate


🧠 First: What is “class imbalance” (in pure English)?

Imagine this situation:

You have 100 patients

95 are healthy

5 have cancer

A very dumb model can say:

“Everyone is healthy”

It will be 95% accurate
…but it misses all cancer cases

That’s class imbalance.

👉 Accuracy looks good
👉 Model is actually bad

🎯 What we are going to study

We will:

Start with balanced data

Slowly remove samples of one class

Retrain the same model

Observe:

Accuracy

Recall (very important here)

🧠 New Metric (explained before code)
🔹 What is Recall?

Recall answers this question:

“Out of all ACTUAL positive cases, how many did the model correctly catch?”

In medical problems:

Recall matters more than accuracy

Missing a disease is worse than a false alarm

You don’t need formulas. Just remember:

Recall = “How many important cases did we catch?”