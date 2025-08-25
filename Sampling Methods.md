# Sampling Methods

> **Summary (TL;DR)**
>
> * You can’t survey everyone, so you pick a **sample** that represents the **population**.
> * **Probability sampling** (randomized) → stronger statistical inference & generalizability.
> * **Non-probability sampling** (non-random) → faster/cheaper but weaker generalization; good for exploratory/qualitative work.
> * Clearly define your **target population**, **sampling frame**, **sample size**, and **bias-reduction** steps in Methods.

# ---

## Key Terms

* **Population:** The full group you want to draw conclusions about.
* **Sample:** The subset you actually study.
* **Sampling frame:** The actual list you sample from (should cover the whole target population, without extras).
* **Sample size (n):** Driven by variability, desired margin of error, confidence level, and study design; state your calculation/assumptions.

# ---

## Probability Sampling (best for quantitative inference)

1. **Simple Random Sampling**
   Everyone has an equal chance; select using a random number generator.
   *Example:* Number employees 1…1000 and randomly pick 100.

2. **Systematic Sampling**
   Select every *k*-th unit from an ordered list (after a random start).
   *Watch out:* Hidden patterns in the list can bias results.
   *Example:* Start at #6, then take 6, 16, 26, …

3. **Stratified Sampling**
   Split population into **strata** (e.g., age, role). Sample each stratum proportionally or equally.
   *Benefit:* Ensures subgroup representation and higher precision.
   *Example:* Sample 80 women and 20 men to match the company’s 80/20 split.

4. **Cluster Sampling**
   Divide into **clusters** that resemble the population (e.g., offices). Randomly pick clusters and survey everyone (or sub-sample) within them.
   *Benefit:* Cost-effective for large, dispersed populations.
   *Risk:* Clusters may differ → higher sampling error.

# ---

## Non-Probability Sampling (exploratory/qualitative, limited generalization)

1. **Convenience Sampling**
   Take who’s easiest to reach. *Fast but biased.*
   *Example:* Survey classmates after your lecture.

2. **Voluntary Response Sampling**
   People opt in (e.g., open online survey). *Self-selection bias likely.*
   *Example:* University-wide survey where only motivated students respond.

3. **Purposive (Judgment) Sampling**
   Use expert judgment to pick cases most useful to the research aim; define inclusion/exclusion criteria.
   *Example:* Select students with varied disability support needs.

4. **Snowball Sampling**
   Ask participants to refer others—useful for hard-to-reach groups.
   *Risk:* Network-driven → representativeness unclear.
   *Example:* Interviewing people experiencing homelessness.

5. **Quota Sampling**
   Non-randomly recruit until subgroup quotas are met.
   *Example:* 200 meat-eaters, 200 vegetarians, 200 vegans for comparison.

# ---

## Choosing a Method (Quick Heuristics)

* Need **representative subgroups** → **Stratified**
* **Geographically dispersed** & budget-limited → **Cluster**
* Have a **complete frame** & want simplicity → **Simple Random** (or **Systematic** if list is safe)
* **Exploratory/qualitative** insight → **Purposive**
* **Hard-to-reach** populations → **Snowball**
* **Fast/cheap pilot** without generalization needs → **Convenience/Voluntary**
* Want **balanced subgroups** without randomness → **Quota**

# ---

## Bias & How to Reduce It

* **Frame coverage:** Ensure the frame fully matches the target population.
* **Systematic patterns:** Check lists for periodic patterns before systematic sampling.
* **Nonresponse:** Use reminders/multiple contacts; report response rates.
* **Weights:** Post-stratify or weight to population margins when appropriate.
* **Pilots:** Dry-run logistics and instruments to catch issues early.

# ---

## What to Report in Methods

* Target **population** and **sampling frame** definitions
* **Sampling method** (why it fits your aim; exactly how applied)
* **Sample size** (calculation, assumptions: variance, margin of error, confidence)
* **Inclusion/exclusion** criteria
* **Response rate**, handling of missing data, and **bias-mitigation** steps
* Any **weights** or **stratum proportions**, plus **limitations**

# ---

## Cheat Sheet

| Method        | Strength                        | Weakness                   | Best Use                                      |
| ------------- | ------------------------------- | -------------------------- | --------------------------------------------- |
| Simple Random | Unbiased, conceptually clean    | Needs full frame           | Small/medium populations with good lists      |
| Systematic    | Easy to implement               | Risk of hidden periodicity | Well-ordered, pattern-free lists              |
| Stratified    | Ensures subgroup representation | Requires good strata info  | Heterogeneous populations; subgroup estimates |
| Cluster       | Cost-effective in the field     | Higher sampling error      | Large, dispersed populations                  |
| Convenience   | Fast, cheap                     | Strong bias                | Pilots, quick checks                          |
| Voluntary     | Easy outreach                   | Self-selection bias        | Open calls, feedback forms                    |
| Purposive     | Deep, targeted insight          | Limited generalization     | Qualitative/specialized phenomena             |
| Snowball      | Access to hidden groups         | Network bias               | Hard-to-reach populations                     |
| Quota         | Balanced subgroups              | Not random                 | Market/survey ops constraints                 |
