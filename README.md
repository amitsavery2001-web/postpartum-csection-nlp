# postpartum-csection-nlp
Style-Invariant Multi-Task Triage of Synthetic Post-C-Section Portal Messages.


# Style-Invariant Multi-Task Triage of Synthetic Post-C-Section Portal Messages

## Project Overview & Clinical Motivation
Following a Cesarean section (C-section), patients recover at home and must self-monitor for potentially life-threatening complications, including surgical site infections (SSIs), uterine hemorrhage, Deep Vein Thrombosis (DVT), and pulmonary embolism (PE). Hospital asynchronous patient portals serve as the primary communication channel, but triage faces two massive bottlenecks:
1. **Linguistic & Stylistic Noise:** Patient portal messages are highly unstructured, informal, emotional, and incomplete. Clinical signal detection is heavily distorted by communication personas exhibiting high anxiety, low medical literacy, or stoic understatement.
2. **The HIPAA Privacy Wall:** Authentic annotated portal data is completely inaccessible to machine learning researchers due to strict privacy and Protected Health Information (PHI) restrictions.

This project addresses these critical limitations by implementing a **controlled, attribute-based synthetic data generation framework** to construct a high-fidelity benchmark dataset. Our primary research objective is to evaluate whether fine-tuned domain-specific transformers and frozen foundational frontier models can achieve style-invariant postpartum medical risk detection.

---

## Formal NLP Task Definition
We frame this objective as a multi-task problem consisting of:
* **Input ($X$):** A raw, variable-length, unstructured free-text patient or caregiver portal message.
* **Task A (Core Benchmark):** Multi-label text classification across 7 post-surgical warning categories + 1 exclusive routine recovery flag:
  `[infection_warning, wound_problem, bleeding_warning, severe_pain_warning, urinary_problem, respiratory_warning, leg_clot_warning, routine_recovery]`
* **Task B (Severity Routing):** Automated ordinal triage level assignment (`Low`, `Medium`, `High`) derived dynamically via a predefined clinical severity mapping rule conditioned on the predicted warning labels from Task A.
* **Task C (Explainability):** Contextual extractive evidence-span isolation (identifying exact text substrings inside $X$ that triggered the warning flags).

---

## Dataset Design & Controlled Generation Pipeline
We will generate a corpus of **3,000 unique synthetic examples** utilizing a structured **Label-to-Data pipeline via GPT-4o**:

1. **Pre-Sampling (Case Plan Matrix):** Before generation, a python script freezes a parameter dictionary determining marginal label frequencies, postpartum timeline buckets, speaker type (first-person patient vs. third-person caregiver), and diversity controls (message length, symptom intensity, vocabulary type, and stylistic noise).
2. **Automated Validation & Rejection Layer:** A post-generation Python script executes 8 structural checks—parsing valid JSON, enforcing exact text substring matching for evidence spans, checking out-of-scope bounds, and filtering out contradictory label assignments. Defective samples are rejected and regenerated.
3. **Data Partitioning:**
   * **Train Split (70%):** Balanced / stratified to maximize clinical signal density.
   * **Validation Split (15%):** Balanced / stratified for hyperparameter and per-label threshold tuning.
   * **Test Set A (15% - Controlled Benchmark):** Balanced / stratified to evaluate base architectural classification limits.
   * **Test Set B (Realistic Clinical Stress Test):** Explicitly imbalanced and routine-heavy to accurately simulate real-world clinical portal deployments.

---

## Modeling & Experimental Benchmarking Matrix
The project implements a side-by-side evaluation across four distinct architectural paradigms under identical seed conditions:
1. **Heuristic Baseline:** TF-IDF n-gram vectorization + independent binary Logistic Regression classifiers (Binary Relevance framework).
2. **General Contextual Encoder:** Full-weight fine-tuning of `RoBERTa-Base` using Binary Cross-Entropy Loss with Logits.
3. **Domain-Specific Medical Encoder:** Full-weight fine-tuning of `Bio_ClinicalBERT` (pre-trained on MIMIC-III clinical notes) to evaluate domain-pretraining performance under severe stylistic distortion.
4. **Frozen Frontier Foundational Models:** Prompt-based benchmarking of `gpt-4o-mini` and `Llama-3-8B-Instruct` across Zero-Shot and Few-Shot configurations (anchors drawn strictly from a prompt-development validation split).

### Core Style Ablation Study
To isolate the effects of stylistic variance, we will conduct an ablation experiment:
* **Configuration 1:** Training models on balanced clinical labels + full style diversity.
* **Configuration 2:** Training models on balanced clinical labels + limited/omitted style diversity.
* Both models will be stress-tested on style-heavy test subsets (High Anxiety, Severe Understatement, Low Literacy) to verify style invariance.

---

## Evaluation Metrics & Quality Control
Models will be thoroughly audited utilizing a dual-test protocol evaluated across:
* **Contextual & Structural Performance:** Macro-F1 (unweighted class average), Micro-F1 (global accuracy balance), and Exact-Match Ratio (strict multi-label full-vector correctness).
* **Clinical Safety Core KPI:** Per-label Recall optimization to minimize missing critical patient signs.
* **High-Risk Miss Rate:** A safety-critical metric calculating the explicit percentage of severe `respiratory_warning` or `leg_clot_warning` indicators that were erroneously predicted as low-risk or routine.

---

##  Project Team & Setup
* **Course:** Final Project in Natural Language Processing (NLP)
* **Status:** Phase 1 (Proposal & Environment Setup)
