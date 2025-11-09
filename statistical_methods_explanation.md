# Statistical Methods Explanation
## Comprehensive Guide to Statistical Analysis in CA2

---

## Table of Contents
1. [Introduction](#introduction)
2. [Hypothesis Testing](#hypothesis-testing)
3. [Correlation Analysis](#correlation-analysis)
4. [Regression Modeling](#regression-modeling)
5. [Assumption Validation](#assumption-validation)
6. [Interpretation Guidelines](#interpretation-guidelines)

---

## Introduction

This document provides detailed explanations of all statistical methods used in the CA2 Statistical Analysis of COVID-19 data. Each method is explained with its purpose, assumptions, interpretation, and practical applications.

### Why Statistical Analysis?

Statistical analysis allows us to:
- **Make informed decisions** based on data rather than intuition
- **Identify patterns** and relationships in complex datasets
- **Test hypotheses** about population parameters using sample data
- **Predict outcomes** based on observed relationships
- **Quantify uncertainty** in our conclusions

---

## Hypothesis Testing

### What is Hypothesis Testing?

Hypothesis testing is a statistical method for making decisions about population parameters based on sample data. It involves:

1. **Formulating hypotheses**: Null (H₀) and alternative (H₁)
2. **Collecting data**: Sample from the population
3. **Calculating test statistic**: Measure of evidence against H₀
4. **Determining p-value**: Probability of observing data if H₀ is true
5. **Making decision**: Reject or fail to reject H₀ based on significance level (α)

### Key Concepts

#### Null Hypothesis (H₀)
- Assumes **no effect** or **no difference** between groups
- Represents the status quo or default position
- Example: "There is no difference in death rates across continents"

#### Alternative Hypothesis (H₁)
- Represents what we're trying to demonstrate
- Contradicts the null hypothesis
- Example: "At least one continent has a different death rate"

#### Significance Level (α)
- Probability threshold for rejecting H₀
- Commonly set at **α = 0.05** (5% risk of Type I error)
- Means we accept 5% chance of incorrectly rejecting H₀

#### P-value
- Probability of obtaining results at least as extreme as observed, assuming H₀ is true
- **p < α**: Reject H₀ (statistically significant)
- **p ≥ α**: Fail to reject H₀ (not statistically significant)

#### Type I and Type II Errors
- **Type I Error (False Positive)**: Rejecting H₀ when it's actually true (controlled by α)
- **Type II Error (False Negative)**: Failing to reject H₀ when it's actually false

---

### 1. ANOVA (Analysis of Variance)

#### Purpose
Compare means of **three or more groups** simultaneously to determine if at least one group mean is significantly different from the others.

#### When to Use
- Comparing a continuous outcome variable across multiple categorical groups
- Example: Comparing death rates across 6 continents

#### Why Not Multiple T-tests?
Performing multiple t-tests increases the **family-wise error rate** (probability of making at least one Type I error). ANOVA controls this by testing all groups simultaneously.

#### How ANOVA Works

ANOVA partitions total variability into:

1. **Between-group variability**: Differences among group means
2. **Within-group variability**: Differences within each group

**F-statistic** = Between-group variance / Within-group variance

- **Large F**: Group means differ more than expected by chance
- **Small F**: Group means similar to what's expected by chance

#### Mathematical Foundation

For k groups with sample sizes n₁, n₂, ..., nₖ:

**Total Sum of Squares (SST)**: 
```
SST = Σ(Xᵢⱼ - X̄)²
```

**Between-Group Sum of Squares (SSB)**:
```
SSB = Σnᵢ(X̄ᵢ - X̄)²
```

**Within-Group Sum of Squares (SSW)**:
```
SSW = Σ(Xᵢⱼ - X̄ᵢ)²
```

**F-statistic**:
```
F = (SSB / (k-1)) / (SSW / (N-k))
```

Where:
- k = number of groups
- N = total sample size
- X̄ᵢ = mean of group i
- X̄ = grand mean

#### Assumptions

1. **Independence**: Observations are independent of each other
2. **Normality**: Data in each group follows approximately normal distribution
3. **Homogeneity of variance**: Variances are equal across groups (homoscedasticity)

**Note**: ANOVA is relatively robust to violations of normality, especially with:
- Large sample sizes (Central Limit Theorem applies)
- Balanced designs (equal sample sizes)

#### Interpreting ANOVA Results

**Example Output**:
```
F-statistic: 8.2456
P-value: 0.000023
Significance level: 0.05
```

**Interpretation**:
- Since p < 0.05, we **reject** the null hypothesis
- Conclusion: At least one continent has a significantly different mean death rate
- **Important**: ANOVA tells us differences exist but not which groups differ

#### Post-Hoc Tests

If ANOVA is significant, use post-hoc tests to identify which specific groups differ:
- **Tukey's HSD**: Controls Type I error rate
- **Bonferroni correction**: More conservative
- **Dunnett's test**: Compares all groups to a control

#### Practical Example from CA2

**Research Question**: Do COVID-19 death rates differ across continents?

**Hypotheses**:
- H₀: μ_Africa = μ_Asia = μ_Europe = μ_North America = μ_South America = μ_Oceania
- H₁: At least one μᵢ is different

**Decision Rule**:
- If p < 0.05: Continental location affects death rates
- If p ≥ 0.05: No evidence of continental differences

---

### 2. Chi-Square Test of Independence

#### Purpose
Determine if there's a **significant association** between two categorical variables.

#### When to Use
- Both variables are categorical (nominal or ordinal)
- Testing independence of variables
- Example: Is continent associated with severity category?

#### Contingency Table

Chi-square test uses a **contingency table** (cross-tabulation):

```
                Low    Medium   High
Africa           30      20      7
Asia             25      15      9
Europe           15      25     17
North America     5      10     24
South America     7       8     11
Oceania           3       2      0
```

#### How Chi-Square Works

Compares **observed frequencies** (O) with **expected frequencies** (E):

**Expected frequency** for cell (i,j):
```
Eᵢⱼ = (Row Total × Column Total) / Grand Total
```

**Chi-square statistic**:
```
χ² = Σ[(Oᵢⱼ - Eᵢⱼ)² / Eᵢⱼ]
```

- **Large χ²**: Observed frequencies differ substantially from expected
- **Small χ²**: Observed frequencies close to what's expected if independent

#### Degrees of Freedom

```
df = (number of rows - 1) × (number of columns - 1)
```

Example: 6 continents × 3 severity levels = (6-1) × (3-1) = 10 df

#### Assumptions

1. **Independence**: Each observation belongs to only one cell
2. **Sample size**: Expected frequency in each cell ≥ 5
   - If violated, consider Fisher's exact test or combine categories
3. **Random sampling**: Data collected randomly from population

#### Interpreting Chi-Square Results

**Example Output**:
```
Chi-square statistic: 45.3217
P-value: 0.000001
Degrees of freedom: 10
```

**Interpretation**:
- Since p < 0.05, we **reject** the null hypothesis
- Conclusion: Significant association between continent and severity
- The distribution of severity categories varies by continent

#### Measuring Strength of Association

**Cramér's V**: Measure of effect size for chi-square test
```
V = √(χ² / (N × min(r-1, c-1)))
```

Where:
- N = total sample size
- r = number of rows
- c = number of columns

Interpretation:
- V = 0.00-0.15: Weak association
- V = 0.15-0.25: Moderate association  
- V > 0.25: Strong association

#### Practical Example from CA2

**Research Question**: Is there an association between continent and COVID-19 severity?

**Hypotheses**:
- H₀: Continent and severity are independent (no association)
- H₁: Continent and severity are associated (dependent)

**Decision Rule**:
- If p < 0.05: Severity distribution differs by continent
- If p ≥ 0.05: Severity distribution is similar across continents

---

## Correlation Analysis

### What is Correlation?

Correlation measures the **strength and direction** of the linear relationship between two continuous variables.

### Pearson Correlation Coefficient (r)

The most common correlation measure, ranging from -1 to +1:

**Formula**:
```
r = Σ[(Xᵢ - X̄)(Yᵢ - Ȳ)] / √[Σ(Xᵢ - X̄)² × Σ(Yᵢ - Ȳ)²]
```

Or equivalently:
```
r = Cov(X, Y) / (σₓ × σᵧ)
```

Where:
- Cov(X, Y) = covariance of X and Y
- σₓ, σᵧ = standard deviations of X and Y

#### Interpreting Correlation Coefficient

**Magnitude (Strength)**:
- |r| = 0.00-0.30: Weak correlation
- |r| = 0.30-0.70: Moderate correlation
- |r| = 0.70-1.00: Strong correlation

**Direction**:
- **r > 0**: Positive correlation (variables increase together)
- **r < 0**: Negative correlation (one increases as other decreases)
- **r = 0**: No linear relationship

**Special Values**:
- r = +1: Perfect positive linear relationship
- r = -1: Perfect negative linear relationship
- r = 0: No linear relationship (but non-linear relationships may exist)

#### Examples

1. **r = 0.95** (TotalCases ↔ TotalDeaths)
   - Very strong positive correlation
   - Countries with more cases tend to have proportionally more deaths

2. **r = -0.35** (TestingRate ↔ MortalityRate)
   - Moderate negative correlation
   - Higher testing rates associated with slightly lower mortality rates

3. **r = 0.05** (Deaths/1M pop ↔ Population)
   - Weak/no correlation
   - Population size doesn't predict per-capita death rate

### Correlation Matrix

A **correlation matrix** shows pairwise correlations between multiple variables:

```
              TotalCases  TotalDeaths  Population  Deaths/1M
TotalCases         1.00         0.95        0.75       0.15
TotalDeaths        0.95         1.00        0.68       0.25
Population         0.75         0.68        1.00      -0.05
Deaths/1M          0.15         0.25       -0.05       1.00
```

**Heatmap Visualization**: Colors represent correlation strength
- Red: Strong positive
- Blue: Strong negative
- White: Weak/no correlation

### Statistical Significance of Correlations

**Test Hypothesis**:
- H₀: ρ = 0 (no correlation in population)
- H₁: ρ ≠ 0 (correlation exists in population)

**T-statistic**:
```
t = r√(n-2) / √(1-r²)
```

With df = n - 2

**P-value**: Probability of observing this correlation if ρ = 0

### Important Considerations

#### 1. Correlation ≠ Causation
- Strong correlation doesn't mean one variable causes the other
- Could be:
  - X causes Y
  - Y causes X
  - Third variable (Z) causes both X and Y
  - Coincidental relationship

#### 2. Linear Relationships Only
- Pearson's r measures **linear** relationships
- May miss non-linear relationships (U-shaped, exponential, etc.)
- Always visualize data with scatter plots

#### 3. Outliers
- Sensitive to outliers (extreme values)
- One outlier can dramatically change correlation coefficient
- Consider robust alternatives (Spearman's rank correlation)

#### 4. Range Restriction
- Correlation weaker when variable range is restricted
- Example: Analyzing only high-income countries reduces variability

### Practical Application in CA2

**Objective**: Identify relationships between COVID-19 variables

**Key Correlations to Explore**:
1. TotalCases ↔ TotalDeaths: Predictive relationship?
2. TestingRate ↔ CaseRate: Does testing affect reported cases?
3. Population ↔ TotalCases: Do larger countries have more cases?
4. Deaths/1M pop ↔ Cases/1M pop: Per-capita relationship?

**Interpretation Example**:
If r(TotalCases, TotalDeaths) = 0.98:
- Very strong positive correlation
- 96% shared variance (r² = 0.96)
- Can predict deaths from cases with high accuracy
- Justifies building regression model

---

## Regression Modeling

### What is Regression Analysis?

Regression models the relationship between:
- **Dependent variable (Y)**: Outcome we want to predict
- **Independent variable(s) (X)**: Predictors

**Purpose**:
- **Prediction**: Estimate Y given X
- **Explanation**: Understand how X affects Y
- **Control**: Account for confounding variables

### Simple Linear Regression

Models relationship between one X and one Y using a straight line.

#### The Regression Equation

```
Y = β₀ + β₁X + ε
```

Where:
- **Y**: Dependent variable (outcome)
- **X**: Independent variable (predictor)
- **β₀**: Intercept (value of Y when X = 0)
- **β₁**: Slope (change in Y per unit change in X)
- **ε**: Error term (unexplained variability)

**Fitted Equation** (estimated from data):
```
Ŷ = b₀ + b₁X
```

Where:
- Ŷ: Predicted value of Y
- b₀: Estimated intercept
- b₁: Estimated slope

#### Estimating Coefficients

**Ordinary Least Squares (OLS)**: Minimize sum of squared residuals

**Slope (b₁)**:
```
b₁ = Σ[(Xᵢ - X̄)(Yᵢ - Ȳ)] / Σ(Xᵢ - X̄)²
   = r × (σᵧ / σₓ)
```

**Intercept (b₀)**:
```
b₀ = Ȳ - b₁X̄
```

#### Interpreting Coefficients

**Example**: Deaths = 150 + 0.011 × Cases

**Intercept (150)**:
- Expected deaths when cases = 0
- Often not meaningful (can't have 0 cases with deaths)
- Mathematical necessity for the line

**Slope (0.011)**:
- For every 1 additional case, expect 0.011 more deaths
- For every 1,000 cases, expect 11 more deaths
- **Key interpretation**: Rate of change in Y per unit change in X

### Model Evaluation Metrics

#### 1. R-squared (R² or Coefficient of Determination)

**Formula**:
```
R² = 1 - (SSresidual / SStotal)
   = SSregression / SStotal
```

Where:
- SSresidual = Σ(Yᵢ - Ŷᵢ)²: Unexplained variation
- SStotal = Σ(Yᵢ - Ȳ)²: Total variation
- SSregression = Σ(Ŷᵢ - Ȳ)²: Explained variation

**Interpretation**:
- Proportion of variance in Y explained by X
- Range: 0 to 1 (0% to 100%)
- R² = 0.85: 85% of variance in deaths explained by cases

**Guidelines**:
- R² > 0.90: Excellent fit
- R² = 0.70-0.90: Good fit
- R² = 0.50-0.70: Moderate fit
- R² < 0.50: Weak fit

**Important Notes**:
- R² always increases with more predictors (in multiple regression)
- Not suitable for comparing models with different numbers of predictors
- Use Adjusted R² for model comparison

#### 2. Adjusted R-squared

**Formula**:
```
R²ₐdⱼ = 1 - [(1 - R²)(n - 1) / (n - p - 1)]
```

Where:
- n = sample size
- p = number of predictors

**Purpose**:
- Penalizes addition of unnecessary predictors
- Only increases if new predictor improves model
- Better for model comparison

#### 3. Root Mean Squared Error (RMSE)

**Formula**:
```
RMSE = √[Σ(Yᵢ - Ŷᵢ)² / n]
```

**Interpretation**:
- Average prediction error in original units
- Lower is better
- RMSE = 5,000: Predictions off by ~5,000 deaths on average

**Advantages**:
- Same units as dependent variable
- Intuitive interpretation
- Penalizes large errors more heavily

#### 4. Mean Absolute Error (MAE)

**Formula**:
```
MAE = Σ|Yᵢ - Ŷᵢ| / n
```

**Interpretation**:
- Average absolute prediction error
- More robust to outliers than RMSE
- MAE = 3,000: Typical prediction off by 3,000 deaths

### Residuals

**Residual** = Observed - Predicted = Yᵢ - Ŷᵢ

**Properties of residuals**:
- Positive residual: Model underpredicted (observed > predicted)
- Negative residual: Model overpredicted (observed < predicted)
- Residual = 0: Perfect prediction

**Uses**:
- Assess model fit
- Check assumptions
- Identify outliers
- Detect patterns in errors

### Prediction

**Point Prediction**: Single predicted value
```
Ŷ = b₀ + b₁X
```

**Prediction Interval**: Range likely to contain actual value
```
Ŷ ± t* × SE(prediction)
```

Where:
- t*: Critical value from t-distribution
- SE(prediction): Standard error of prediction

**Example**:
- For X = 100,000 cases: Ŷ = 1,250 deaths
- 95% PI: (1,100, 1,400) deaths
- Interpretation: 95% confident actual deaths will be between 1,100 and 1,400

### Multiple Regression (Extension)

When multiple predictors (X₁, X₂, ..., Xₚ):

```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε
```

**Advantages**:
- Account for multiple factors simultaneously
- Control for confounding variables
- More accurate predictions

**Coefficient Interpretation**:
- βᵢ: Change in Y per unit change in Xᵢ, **holding all other predictors constant**
- Example: β₁ = 0.008 means 0.008 more deaths per case, controlling for population

### Practical Example from CA2

#### Model 1: Total Deaths ~ Total Cases

**Equation**: Deaths = 150.00 + 0.010912 × Cases

**Interpretation**:
- Intercept: Baseline deaths when no cases (not meaningful)
- Slope: ~0.011 deaths per case, or 11 deaths per 1,000 cases
- This represents the case fatality relationship

**R² = 0.9523**:
- 95.23% of variance in deaths explained by cases
- Excellent predictive power
- Strong evidence of systematic relationship

**Application**:
- Predict: For 50,000 cases → Deaths = 150 + 0.010912(50,000) = 696 deaths
- Policy: Can forecast healthcare needs based on case projections

#### Model 2: Deaths/1M pop ~ Cases/1M pop

**Why per-capita analysis**:
- Accounts for population size differences
- Fair comparison across countries
- Measures intensity rather than absolute numbers

**Interpretation**:
- Slope: Death rate change per unit change in case rate
- More relevant for public health comparisons
- Isolates disease impact from population effects

---

## Assumption Validation

Statistical tests have **assumptions** that must be checked for valid results. Violating assumptions can lead to:
- Invalid p-values
- Biased estimates
- Incorrect conclusions

### ANOVA Assumptions

#### 1. Independence

**Assumption**: Observations are independent of each other

**How to check**:
- Study design: Random sampling, no repeated measures
- Domain knowledge: No obvious dependencies

**Violation consequences**:
- Inflated Type I error rate
- P-values too small (overconfident)

**Solutions**:
- Use repeated measures ANOVA for dependent data
- Account for clustering in analysis

#### 2. Normality

**Assumption**: Data within each group follows normal distribution

**How to check**:

a) **Visual methods**:
- **Q-Q plot**: Points should fall along diagonal line
- **Histogram**: Should be bell-shaped
- **Boxplot**: Symmetric, few outliers

b) **Statistical tests**:
- **Shapiro-Wilk test**:
  - H₀: Data is normally distributed
  - H₁: Data is not normally distributed
  - p > 0.05: Consistent with normality
  - p < 0.05: Significant deviation from normality

**Interpretation**:
```
W-statistic: 0.945
P-value: 0.234
```
- Not significant (p > 0.05)
- No strong evidence against normality

**Violation consequences**:
- ANOVA relatively robust with large samples (Central Limit Theorem)
- More problematic with small, unequal sample sizes

**Solutions**:
- Transform data (log, sqrt)
- Use non-parametric alternative (Kruskal-Wallis test)
- Proceed with caution if n > 30 per group

#### 3. Homogeneity of Variance (Homoscedasticity)

**Assumption**: Variances are equal across groups

**How to check**:

a) **Visual methods**:
- **Boxplots**: Similar box heights
- **Residual plots**: Equal spread across groups

b) **Statistical tests**:
- **Levene's test**:
  - H₀: Variances are equal
  - H₁: Variances are not equal
  - p > 0.05: Homogeneity assumption met
  - p < 0.05: Variances differ significantly

**Interpretation**:
```
Levene's statistic: 2.345
P-value: 0.087
```
- Not significant (p > 0.05)
- Variances approximately equal

**Violation consequences**:
- Inflated Type I error (unequal sample sizes)
- Reduced power

**Solutions**:
- **Welch's ANOVA**: Doesn't assume equal variances
- Transform data to stabilize variance
- Use robust statistics

### Regression Assumptions

#### 1. Linearity

**Assumption**: Relationship between X and Y is linear

**How to check**:
- **Scatter plot**: Points should follow linear pattern
- **Residual plot**: No curved pattern in residuals vs. fitted values

**Violation consequences**:
- Biased predictions
- Underestimated R²
- Misleading coefficients

**Solutions**:
- Transform variables (log, polynomial)
- Use non-linear regression models
- Add interaction terms or polynomial terms

#### 2. Independence of Errors

**Assumption**: Residuals are independent

**How to check**:
- **Residual plot**: No patterns over time or order
- **Durbin-Watson test**: Tests for autocorrelation
  - DW ≈ 2: No autocorrelation
  - DW < 2: Positive autocorrelation
  - DW > 2: Negative autocorrelation

**Violation consequences**:
- Underestimated standard errors
- Overconfident hypothesis tests
- Invalid confidence intervals

**Solutions**:
- Time series methods
- Include lagged variables
- Use autoregressive models

#### 3. Homoscedasticity (Constant Variance)

**Assumption**: Residuals have constant variance across all levels of X

**How to check**:
- **Residual plot**: Should show constant spread (no funnel shape)
- **Scale-location plot**: Horizontal line with equal spread

**Patterns indicating violations**:
- Funnel shape: Variance increases with X
- Diamond shape: Variance highest in middle

**Violation consequences**:
- Inefficient estimates (not minimum variance)
- Incorrect standard errors
- Invalid hypothesis tests

**Solutions**:
- Transform dependent variable (log, sqrt)
- Use Weighted Least Squares (WLS)
- Use robust standard errors

#### 4. Normality of Residuals

**Assumption**: Residuals follow normal distribution

**How to check**:
- **Q-Q plot**: Residuals vs. theoretical quantiles should be linear
- **Histogram**: Should be bell-shaped
- **Shapiro-Wilk test**: Applied to residuals

**Violation consequences**:
- Invalid confidence intervals (small samples)
- Less efficient estimates
- Regression relatively robust to violations

**Solutions**:
- Transform dependent variable
- Use robust regression
- Bootstrap confidence intervals

#### 5. No Outliers or Influential Points

**How to identify**:
- **Residual plots**: Points far from others
- **Leverage**: How much X value differs from X̄
- **Cook's Distance**: Combined influence on regression line
  - D > 1: Potentially influential
  - D > 4/n: Worth investigating

**Actions**:
- Investigate: Data entry error? Valid observation?
- Analyze with and without outliers
- Use robust regression methods

### Diagnostic Plots

#### Residuals vs. Fitted Values
**Purpose**: Check linearity and homoscedasticity

**Good pattern**:
- Random scatter around zero
- No discernible pattern
- Constant spread

**Bad patterns**:
- Curved: Non-linearity
- Funnel: Heteroscedasticity
- Clusters: Missing variables

#### Q-Q Plot
**Purpose**: Check normality of residuals

**Good pattern**:
- Points fall along diagonal line
- Few deviations at extremes OK

**Bad patterns**:
- S-shaped: Heavy tails
- Curved: Skewness
- Jumps: Discrete data

#### Scale-Location Plot
**Purpose**: Check homoscedasticity

**Good pattern**:
- Horizontal line
- Equal vertical spread

**Bad patterns**:
- Increasing trend: Growing variance
- Decreasing trend: Shrinking variance

#### Residuals vs. Leverage
**Purpose**: Identify influential observations

**Focus areas**:
- High leverage + large residual = Very influential
- Cook's distance contours show influence regions

---

## Interpretation Guidelines

### Statistical Significance vs. Practical Significance

**Statistical Significance**:
- p < α: Result unlikely due to chance
- Depends heavily on sample size
- Large samples: Even tiny effects are "significant"

**Practical Significance**:
- Is the effect large enough to matter?
- Does it have real-world implications?
- Cost-benefit considerations

**Example**:
- p = 0.001: Statistically significant ✓
- Effect: 0.5 deaths per million
- Practical significance: Minimal impact on policy ✗

### Effect Size

Quantifies the magnitude of an effect, independent of sample size.

**Common measures**:
- **Cohen's d** (t-test): Standardized mean difference
  - Small: d = 0.2
  - Medium: d = 0.5
  - Large: d = 0.8

- **η² (eta-squared)** (ANOVA): Proportion of variance explained
  - Small: η² = 0.01
  - Medium: η² = 0.06
  - Large: η² = 0.14

- **Cramér's V** (Chi-square): Association strength
  - Small: V = 0.10
  - Medium: V = 0.30
  - Large: V = 0.50

- **R²** (Regression): Variance explained
  - Small: R² = 0.02
  - Medium: R² = 0.13
  - Large: R² = 0.26

### Confidence Intervals

Provide a range of plausible values for a parameter.

**95% Confidence Interval**:
- If we repeated the study many times, 95% of intervals would contain the true parameter
- **NOT**: 95% probability the parameter is in this interval

**Interpretation**:
```
β₁ = 0.011, 95% CI: (0.009, 0.013)
```
- Point estimate: 0.011 deaths per case
- Range: Between 0.009 and 0.013 deaths per case (95% confidence)
- Interval doesn't include 0 → Significant at α = 0.05

**Advantages**:
- Shows precision of estimate
- Provides context for p-value
- Indicates practical significance

### Reporting Statistical Results

**Complete reporting includes**:

1. **Descriptive statistics**:
   - Sample sizes
   - Means and standard deviations
   - Medians and ranges

2. **Test statistics**:
   - Name of test used
   - Test statistic value (F, χ², t, etc.)
   - Degrees of freedom
   - P-value
   - Effect size

3. **Assumptions**:
   - Which assumptions were checked
   - Results of assumption tests
   - Any violations and how addressed

4. **Interpretation**:
   - Statistical conclusion
   - Practical implications
   - Limitations

**Example**:
> A one-way ANOVA was conducted to compare death rates across continents. The assumption of homogeneity of variance was met (Levene's test: F(5, 217) = 1.86, p = 0.102). Death rates differed significantly across continents, F(5, 217) = 8.25, p < 0.001, η² = 0.16. Europe (M = 1,850, SD = 890) had significantly higher death rates than Asia (M = 890, SD = 720), p = 0.003. This represents a large and practically significant difference.

### Common Pitfalls to Avoid

1. **P-hacking**: Trying multiple tests until finding p < 0.05
2. **HARKing**: Hypothesizing After Results are Known
3. **Ignoring assumptions**: Proceeding despite violations
4. **Confusing correlation with causation**
5. **Over-interpreting non-significant results**: Absence of evidence ≠ evidence of absence
6. **Cherry-picking results**: Only reporting favorable findings
7. **Ignoring effect size**: Focusing only on p-values
8. **Multiple comparisons**: Not adjusting α for multiple tests

---

## Comprehensive Example: Putting It All Together

### Research Scenario
Analyzing COVID-19 death rates across continents

### Step 1: Descriptive Analysis
- Calculate means, SDs, medians for each continent
- Create visualizations (boxplots, histograms)
- Identify outliers and missing data

### Step 2: Hypothesis Testing (ANOVA)
**Hypotheses**:
- H₀: All continent means are equal
- H₁: At least one continent differs

**Assumptions**:
- Check normality (Shapiro-Wilk per group)
- Check homogeneity (Levene's test)
- Verify independence (study design)

**Results**:
- F(5, 217) = 8.25, p < 0.001, η² = 0.16
- Reject H₀: Continents differ significantly

**Post-hoc**:
- Tukey's HSD to identify which continents differ

### Step 3: Association Analysis (Chi-Square)
**Hypotheses**:
- H₀: Continent and severity are independent
- H₁: Continent and severity are associated

**Results**:
- χ²(10) = 45.32, p < 0.001, V = 0.32
- Reject H₀: Significant association

**Interpretation**:
- Severity distribution varies by continent
- Medium effect size (V = 0.32)

### Step 4: Correlation Analysis
**Key correlations**:
- TotalCases ↔ TotalDeaths: r = 0.98, p < 0.001
- Deaths/1M ↔ Cases/1M: r = 0.65, p < 0.001

**Interpretation**:
- Very strong relationship between cases and deaths
- Justifies building predictive regression model

### Step 5: Regression Modeling
**Model**: Deaths = β₀ + β₁ × Cases

**Results**:
- b₀ = 150.00, b₁ = 0.0109
- R² = 0.95, p < 0.001
- RMSE = 45,000 deaths

**Assumptions**:
- Linearity: ✓ (scatter plot linear)
- Normality: ✓ (Q-Q plot good)
- Homoscedasticity: ✓ (residual plot constant spread)
- Independence: ✓ (cross-sectional data)

**Interpretation**:
- For every 1,000 cases, expect ~11 deaths
- Model explains 95% of variance
- Excellent predictive accuracy

### Step 6: Conclusions
**Statistical**:
- Significant continental differences exist
- Strong predictive relationship established
- Association between location and severity

**Practical**:
- Different continents need tailored interventions
- Can forecast deaths from case projections
- Resource allocation should account for geography

**Limitations**:
- Cross-sectional (no causation)
- Data quality varies by country
- Confounding variables not controlled

---

## Summary

This document has provided comprehensive explanations of:

1. **Hypothesis Testing**: ANOVA and Chi-square tests for comparing groups and testing associations
2. **Correlation Analysis**: Measuring strength and direction of relationships
3. **Regression Modeling**: Predicting outcomes and understanding relationships
4. **Assumption Validation**: Ensuring valid statistical inferences
5. **Interpretation Guidelines**: Moving from statistics to practical insights

### Key Takeaways

- **Always check assumptions** before interpreting results
- **Effect size matters** as much as statistical significance
- **Visualize data** before and after analysis
- **Report completely**: test statistics, assumptions, limitations
- **Think practically**: Statistical significance ≠ practical importance
- **Be cautious**: Correlation ≠ causation, p-values can mislead

### Further Learning

To deepen your understanding:
1. Practice with different datasets
2. Study assumption violations and remedies
3. Learn advanced methods (mixed models, Bayesian analysis)
4. Read statistical reporting guidelines (APA, CONSORT)
5. Understand the philosophy of statistical inference

---

## References and Resources

### Textbooks
- Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics* (5th ed.)
- Agresti, A. (2018). *Statistical Methods for the Social Sciences* (5th ed.)
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*

### Online Resources
- Khan Academy: Statistics and Probability
- StatQuest: YouTube channel with intuitive explanations
- CrossValidated: Q&A site for statistics

### Software Documentation
- SciPy documentation: Statistical functions in Python
- Statsmodels documentation: Statistical modeling in Python
- Scikit-learn documentation: Machine learning in Python

---

*Document prepared for CA2: Statistical Modelling & Inference*  
*Last updated: November 2025*
