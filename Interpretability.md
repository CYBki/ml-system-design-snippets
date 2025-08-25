# Model Interpretability

## Why interpretability?
- **Metrics aren’t enough:** The training objective (e.g., accuracy) often fails to capture real-world **ethics, fairness, risk, and cost**.
- **Interpretability** helps you see **what the model learned**, **why it made a decision**, and whether you can **trust** it in deployment.
- Critical when stakes are high (health/finance), when there’s distribution shift, or when regulation requires explanations.

---

## What does “interpretability” mean?
Based on **Lipton (2016)** and **Doshi-Velez & Kim (2017)**.

### Two broad families
1) **Transparency** — understanding the **mechanism**:
   - **Simulatability:** A human could reproduce the computation.
   - **Decomposability:** Inputs/parameters/operations each have intuitive meaning.
   - **Algorithmic transparency:** Guarantees about optimization/behavior.  
   *Trade-off:* usually pushes you toward **simple** models (linear, shallow trees).

2) **Post-hoc explanations** — extracting information **after** training:
   - Textual rationales, visualizations, **local** explanations (saliency), example-based explanations.  
   *Powerful but may describe outputs, not the true mechanism.*

### What do we want from interpretability?
- **Trust:** Knowing **where** the model fails and how it behaves in production.
- **Causality:** Hypothesis generation for real-world testing.
- **Transferability:** Understanding behavior under shift/adversarial attacks.
- **Informativeness:** Extra signals/features for human decision-makers.
- **Fair/ethical decisions:** Explanations that are contestable and correctable.

---

## Methods

### 1) **LIME** (local, model-agnostic)
- **Idea:** Fit a simple local surrogate (e.g., sparse linear) around each instance to attribute feature effects.
- **Pros:** Works with any black box; great for **debugging** and judging **single predictions**; SP-LIME picks a small representative set.
- **Cons:** Local fidelity/consistency not guaranteed; requires **human-interpretable features** (superpixels, BoW); can be misleading.

### 2) **SHAP** (Shapley values for feature attribution)
- **Idea:** Compute feature attributions that satisfy **local accuracy, missingness, and consistency**.
- **Tooling:**  
  - **KernelExplainer** (model-agnostic; like a principled LIME, often slower)  
  - **TreeExplainer** (fast for tree/boosting, supports global importance; may assume feature independence for some transforms)  
  - **DeepExplainer** (DNNs; faster than Kernel, still costly at scale)
- **Pros:** Strong **axiomatic guarantees**; local + **global** analysis (importance ranks, dependence plots).
- **Cons:** Still **post-hoc**; speed/assumption trade-offs (esp. Kernel/Tree independence).

### 3) **CNN Visualization (Olah et al.)**
- **Idea:** Visualize/aggregate **hidden-layer activations** (neurons/channels/spatial maps) and factorize them into **human-scale** components to see what the network actually represents.
- **Pros:** Peeks **inside the mechanism**; shows intermediate structures driving the output.
- **Cons:** Harder to digest for non-experts; potential observer bias; no simulatability/optimization guarantees.

---

## How to evaluate interpretability (Doshi-Velez & Kim)
1) **Application-grounded:** Expert users on real tasks.  
2) **Human-grounded:** Non-experts on simplified tasks.  
3) **Functionally-grounded:** No humans; proxy metrics (sparsity, local fidelity).  
*Pick the level that matches your claim and context; they should inform each other.*

---

## Which method when?

| Need | Best fit |
|---|---|
| Explain a **single decision** quickly, model-agnostic debugging | **LIME** (or SHAP Kernel if you can afford the cost) |
| **Local + global** explanations with **consistent** attributions, especially for tree ensembles | **SHAP (TreeExplainer)** |
| Understand what a deep net **internally represents** (research/oversight) | **CNN visualization (Olah)** |
| Compliance/fairness reports with simple, stable narratives | **SHAP** |
| Detect spurious correlations / distribution shift | LIME/SHAP (local checks) + CNN visualizations for “what the model looks at” |

### Common pitfalls
- Post-hoc explanations can be **persuasive but wrong**; don’t confuse plausibility with truth.  
- If features aren’t human-interpretable, explanations won’t be either—fix representation first.  
- Global claims need more than a handful of local explanations.  
- TreeExplainer’s independence assumption can mislead under correlated features.

---

## Takeaway
- Interpretability isn’t one thing; clarify **which goal** (trust, fairness, transfer, insight) you need.
- **LIME/SHAP** explain decisions via **feature attributions** (post-hoc); **Olah** exposes **internal representations** of deep nets.
- Use the framework to choose consciously: *“Use SHAP because its decomposability benefits outweigh post-hoc risks for our stakeholders,”* not just *“use SHAP.”*
