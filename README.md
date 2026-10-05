# Predicting Mortgage Denials from Public HMDA Data

**DTSC 2301 · Personal Portfolio Project Two**

In this project I built a machine learning model that predicts whether a home loan application in North Carolina gets **Denied** or **Approved**. I only used information that lenders have to report to the public. I also checked whether the model makes more mistakes for some groups of people than for others.

📓 **Full notebook:** [`Project 2 Workbook.ipynb`](Project_2_Workbook.ipynb)

---

## The Question

> Using only the information lenders publicly report, how accurately can I predict whether a conventional, owner-occupied home-purchase loan application in North Carolina is denied?

I also asked two smaller questions:
1. Which features matter most for predicting a denial?
2. Does the model make more mistakes for some racial, ethnic, or sex groups, even though it never sees those fields?

## Why It Matters

For most families, buying a home is the biggest money decision they will ever make, and a denied loan can delay or end that plan. The public data leaves out credit scores, which lenders rely on heavily. So I knew I wouldn't be able to copy a lender's full decision, I just wanted to see how much of the outcome public data can explain, and whether the model's errors are fair across groups.

## The Data

- **Source:** 2025 Home Mortgage Disclosure Act (HMDA) data for North Carolina, from the [FFIEC HMDA Data Browser](https://ffiec.cfpb.gov/data-browser/)
- **One row = one loan application**
- **Raw size:** 552,230 applications and 99 columns
- **After filtering and cleaning:** 95,270 applications
- **Target:** `denied` (1 = denied, 0 = approved)
- **Denial rate:** 14.8%, so only about 1 in 7 applications is denied

I narrowed the data to applications that reached a final decision and were for conventional, first-lien, owner-occupied home purchases. Loans like refinances, FHA/VA loans, and investment properties follow different rules, so mixing them in would blur the results.

## How I Built It

### 1. Cleaning the data
- I found that missing values were **not random**. Applications missing their loan-to-value ratio were denied **72.8%** of the time, compared to **7.5%** when it was present. If I had just deleted every row with a missing value, I would have thrown out **61% of all denials**.
- Because of that, I calculated loan-to-value myself (loan amount ÷ property value) instead of using the reported column.
- I converted HMDA's hidden missing codes (like `"Exempt"`) into real missing values, turned text columns into numbers, and capped extreme loan-to-value values at 150%.

### 2. Checking for data leakage
Data leakage happens when a feature secretly gives away the answer. I checked every feature and found two leaks:
- **`interest_rate`** was missing for 100% of denied applications, since only approved loans get a rate. I removed it.
- **`preapproval`** had a 0% denial rate. That turned out to be a side effect of my own filtering: denied preapprovals have a different code that I had already excluded. I removed it too.

### 3. Features I used (8)
Debt-to-income ratio, loan-to-value ratio, income, loan amount, property value, loan term, construction method (site-built vs. manufactured home), and applicant age.

Race, ethnicity, and sex were **never** used as inputs. I only used them afterward to check the model for fairness.

### 4. Models
- **Baseline:** a simple rule that predicts "denied" when debt-to-income is 50% or higher, which is the usual limit for conventional loans
- **Model 1:** Logistic Regression
- **Model 2:** Gradient Boosting (tuned with 5-fold cross-validation)

I used an 80/20 train/test split that kept the same denial rate in both sets. Both models used the same split and the same settings for handling the imbalanced classes, so the comparison was fair.

## Results

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Baseline (DTI ≥ 50%) | 0.403 | **0.815** | 0.540 | 0.694 | 0.493 |
| Logistic Regression | 0.766 | 0.645 | 0.700 | 0.878 | 0.704 |
| **Gradient Boosting** | **0.809** | 0.679 | **0.738** | **0.918** | **0.802** |

**I chose Gradient Boosting.** It scored best on 4 of the 5 metrics, including PR-AUC and F1, which are the most important ones when the classes are this uneven. It caught **81% of denials**. Compared to logistic regression, it had both fewer missed denials (538 vs. 659) and fewer false alarms (1,079 vs. 1,189).

The baseline rule had the highest precision, but only because it catches the most obvious cases. It missed 60% of all denials.

## What I Found

**Debt-to-income ratio matters most by far.** When I shuffled it to test its importance, the model's PR-AUC dropped by 0.351, more than four times the next feature. Denial rates jumped from around 10% to 82% once debt-to-income passed about 50%.

**Manufactured homes are a big part of the story.** Manufactured homes were denied 65% of the time, compared to about 6% for site-built homes. Several features (low income, low property value, 300-month loan terms) mostly point to this same group of loans.

## Fairness Audit

Even without seeing race, the model's mistakes were **not** spread evenly:
- Approved **Black** applicants were wrongly flagged as denied about **twice as often** as approved **White** applicants (12.2% vs. 6.4%).
- **American Indian / Alaska Native** applicants had the highest false-alarm rate (23.4%), but this group was small (257 applications), so the number is less certain.
- The model missed more actual denials for **Asian** applicants (it caught 63.5% vs. 78.2% for White applicants).

This probably happens because features like debt-to-income and property type are linked to race in this data, so they act as stand-ins for it. **This does not prove that lenders discriminated.** The public data is missing credit scores and other information lenders use, and research using that fuller data (Bhutta et al., 2025) found that those factors explain most of the gap in denial rates.

## Limitations

- No credit scores, which is a major part of real lending decisions
- Only one state and one year (North Carolina, 2025)
- Rows I dropped for missing values had higher-than-average denial rates
- The results show patterns, not causes

**This model should not be used to make real lending decisions.** It's meant for studying patterns in public data.

## Sources

- Babaei, G., Giudici, P., & Wu, L. (2026). Explainable fairness in mortgage lending. In R. Guidotti, U. Schmid, & L. Longo (Eds.), *Explainable artificial intelligence: Third World Conference, xAI 2025, Istanbul, Turkey, July 9–11, 2025, proceedings, Part IV* (pp. 378–398). Springer. https://doi.org/10.1007/978-3-032-08330-2_18
- Bhutta, N., Hizmo, A., & Ringo, D. (2025). How much does racial bias affect mortgage lending? Evidence from human and algorithmic credit decisions. *The Journal of Finance, 80*(3), 1463–1496. https://doi.org/10.1111/jofi.13444
- Federal Financial Institutions Examination Council. (n.d.). *HMDA data browser* [Data set]. Retrieved October 4, 2026, from https://ffiec.cfpb.gov/data-browser/
- Federal Financial Institutions Examination Council. (n.d.). *Public HMDA – LAR data fields*. https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields
- Zou, L., & Khern-am-nuai, W. (2023). AI and housing discrimination: The case of mortgage applications. *AI and Ethics, 3*(4), 1271–1281. https://doi.org/10.1007/s43681-022-00234-9

# **AI Transparency**

I used AI to write the code of the visualizations to save time on that end, and also used it help me find and fix bugs in my data analysis and cleaning code cells. However all the decisions pertaining towards which charts and graphs I wanted and the data cleaning decisions alongside the explanation as to why behind was driven by me.

In addition, I used AI to clean up the messy notebook and turned it into an notebook that looks good for presentation and made it easy to read and follow from top to bottom.
