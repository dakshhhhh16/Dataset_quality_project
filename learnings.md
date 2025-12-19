# Dataset Quality Project - Key Learnings

## 🎯 Project Focus

### Your project is **NOT** about:
- ❌ Building the best cancer model
- ❌ Beating benchmarks
- ❌ Optimizing hyperparameters

### Your project **IS** about:
- ✅ **Proving that data quality controls model behavior more than model choice**

---

## 📊 Basic ML Model Pipeline

```
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
```

---

## 🧠 Understanding Class Imbalance

### What is "class imbalance"?  (in plain English)

Imagine this situation: 

- You have **100 patients**
  - 95 are healthy
  - 5 have cancer

A very dumb model can say: 
> *"Everyone is healthy"*

**Result:**
- ✅ It will be **95% accurate**
- ❌ But it **misses all cancer cases**

**That's class imbalance.**

#### The Problem: 
- 👉 Accuracy looks good
- 👉 Model is actually bad

---

## 🔬 What We Are Going to Study

### Experimental Approach:

1. **Start** with balanced data
2. **Slowly remove** samples of one class
3. **Retrain** the same model
4. **Observe** the following metrics:
   - Accuracy
   - **Recall** *(very important here)*

---

## 📈 Understanding Metrics

### 🔹 What is Recall? 

**Recall answers this question:**
> *"Out of all ACTUAL positive cases, how many did the model correctly catch?"*

#### Why Recall Matters in Medical Problems: 

| Metric | Importance |
|--------|------------|
| **Recall** | 🔴 **Critical** - Missing a disease is worse than a false alarm |
| Accuracy | ⚪ Can be misleading with imbalanced data |

#### Simple Definition: 

> **Recall = "How many important cases did we catch?"**

💡 *You don't need formulas. Just remember this concept! *

---

## 📝 Summary

This project demonstrates how **data quality** (specifically class balance) directly impacts model performance - regardless of which algorithm you choose.  By systematically reducing samples from one class, we can observe how metrics like recall deteriorate, proving that **data quality > model complexity**. 
