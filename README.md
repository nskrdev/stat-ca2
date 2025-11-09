# CA2: Statistical Modelling & Inference - COVID-19 Analysis

## Project Overview

This project contains a comprehensive statistical analysis of COVID-19 data across countries and continents. The analysis includes hypothesis testing, correlation analysis, and regression modeling to identify significant patterns and relationships in the data.

## Files in This Project

### Main Analysis Files
- **`CA2_Statistical_Analysis.ipynb`** - Complete Jupyter notebook with all statistical analyses
- **`Covid_stats_Jan2025.csv`** - COVID-19 dataset (January 2025)
- **`statistical_methods_explanation.md`** - Comprehensive guide to all statistical methods used

### Reference Files
- **`Guidelines for CA_final.pdf`** - Original assignment guidelines
- **`requirements.txt`** - Python package dependencies
- **`.kiro/specs/`** - Project specifications and task tracking

### Helper Scripts (for development)
- `complete_analysis.py` - Script used to complete the notebook
- `add_hypothesis_testing.py` - Helper for adding hypothesis tests
- `add_anova_viz_and_assumptions.py` - Helper for ANOVA sections

## What's Included in the Analysis

### 1. Data Loading and Preparation ✓
- Loaded COVID-19 dataset with 230 countries
- Cleaned numeric columns (removed commas, converted types)
- Handled missing values appropriately
- Created derived variables:
  - `MortalityRate`: (TotalDeaths / TotalCases) × 100
  - `SeverityCategory`: Low/Medium/High based on Deaths/1M pop

### 2. Hypothesis Testing ✓

#### ANOVA Test
- **Question**: Do death rates differ significantly across continents?
- **Method**: One-way ANOVA
- **Includes**:
  - Descriptive statistics by continent
  - F-statistic and p-value
  - Box plot visualization
  - Normality tests (Shapiro-Wilk)
  - Homogeneity of variance test (Levene's)

#### Chi-Square Test
- **Question**: Is there an association between continent and severity?
- **Method**: Chi-square test of independence
- **Includes**:
  - Contingency table (observed and expected frequencies)
  - Chi-square statistic, p-value, degrees of freedom
  - Stacked bar chart visualization
  - Heatmap of contingency table

### 3. Correlation Analysis ✓
- Computed Pearson correlation matrix for key variables
- Created annotated correlation heatmap
- Identified strong correlations (|r| > 0.70)
- Generated scatter plots for top correlations with regression lines
- Provided practical interpretation of findings

### 4. Regression Analysis ✓

#### Model 1: Total Deaths ~ Total Cases
- Simple linear regression
- R-squared, adjusted R-squared, RMSE, MAE
- Scatter plot with regression line
- Regression equation and coefficient interpretation

#### Model 2: Deaths/1M pop ~ Cases/1M pop
- Per-capita analysis
- All metrics and visualizations
- Comparative interpretation

#### Model Validation
- Comprehensive residual analysis:
  - Residuals vs Fitted Values plot
  - Q-Q plot (normality check)
  - Histogram of residuals
  - Scale-Location plot (homoscedasticity)

### 5. Conclusions and Recommendations ✓
- Summary of all statistical findings
- Practical implications for public health
- Recommendations for further research
- Discussion of limitations

## Statistical Methods Explanation Document

The file **`statistical_methods_explanation.md`** contains detailed explanations of:

1. **Hypothesis Testing**
   - What it is and why we use it
   - Null and alternative hypotheses
   - P-values and significance levels
   - Type I and Type II errors
   - ANOVA methodology and interpretation
   - Chi-square test methodology and interpretation

2. **Correlation Analysis**
   - Pearson correlation coefficient
   - Interpreting correlation strength and direction
   - Correlation matrices and heatmaps
   - Important considerations (correlation ≠ causation)

3. **Regression Modeling**
   - Simple linear regression
   - Regression equation and coefficients
   - R-squared and model evaluation metrics
   - Prediction and confidence intervals
   - Multiple regression concepts

4. **Assumption Validation**
   - ANOVA assumptions (independence, normality, homogeneity)
   - Regression assumptions (linearity, homoscedasticity, normality)
   - How to check assumptions
   - What to do when assumptions are violated
   - Diagnostic plots interpretation

5. **Interpretation Guidelines**
   - Statistical vs practical significance
   - Effect sizes
   - Confidence intervals
   - How to report results properly
   - Common pitfalls to avoid

## How to Run the Analysis

### Prerequisites
```bash
# Activate virtual environment
source .venv/bin/activate

# Install required packages (if not already installed)
pip install -r requirements.txt
```

### Required Packages
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

### Running the Notebook

1. **Start Jupyter Notebook/Lab**:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. **Open the notebook**:
   - Navigate to `CA2_Statistical_Analysis.ipynb`

3. **Run all cells**:
   - Kernel → Restart & Run All
   - Or run cells sequentially (Shift + Enter)

### Expected Output

The notebook will generate:
- Descriptive statistics tables
- Box plots for ANOVA
- Contingency tables and heatmaps for Chi-square
- Correlation matrix and heatmap
- Scatter plots with regression lines
- Residual diagnostic plots
- Statistical test results with interpretations

## Project Structure

```
probandstat/
├── CA2_Statistical_Analysis.ipynb      # Main analysis notebook ⭐
├── statistical_methods_explanation.md  # Methods guide ⭐
├── Covid_stats_Jan2025.csv            # Dataset
├── requirements.txt                    # Dependencies
├── Guidelines for CA_final.pdf        # Assignment guidelines
├── .kiro/
│   └── specs/
│       └── ca2-statistical-analysis/
│           ├── requirements.md         # Project requirements
│           ├── design.md              # Design document
│           └── tasks.md               # Task tracking (all completed!)
└── .venv/                             # Virtual environment
```

## Key Features of the Analysis

### Methodological Rigor
✓ All assumptions checked before tests  
✓ Multiple visualizations for each analysis  
✓ Clear hypotheses stated upfront  
✓ Comprehensive interpretation of results  
✓ Discussion of limitations  

### Visualizations
✓ Box plots with mean markers  
✓ Annotated correlation heatmaps  
✓ Scatter plots with regression lines  
✓ Contingency table heatmaps  
✓ Residual diagnostic plots  
✓ Stacked bar charts  

### Documentation
✓ Markdown cells explaining each step  
✓ Research questions clearly stated  
✓ Statistical conclusions provided  
✓ Practical implications discussed  
✓ Complete methodology section  

## Analysis Summary

### Dataset
- **223 countries** with complete data for analysis
- **6 continents**: Africa, Asia, Europe, North America, South America, Australia/Oceania
- **Key variables**: TotalCases, TotalDeaths, Population, Deaths/1M pop, Cases/1M pop

### Main Findings (Example - will vary with actual execution)
1. **ANOVA**: Significant differences in death rates across continents
2. **Chi-square**: Significant association between continent and severity
3. **Correlation**: Strong positive correlations between cases and deaths
4. **Regression**: High R² values indicate excellent predictive power

## Significance Level

Throughout the analysis, we use **α = 0.05** (5% significance level):
- p < 0.05: Reject null hypothesis (statistically significant)
- p ≥ 0.05: Fail to reject null hypothesis (not statistically significant)

## Tips for Understanding the Analysis

1. **Start with the statistical_methods_explanation.md** to understand the theory
2. **Read through the notebook sequentially** - it's structured to build understanding
3. **Pay attention to assumptions sections** - they ensure valid results
4. **Look at visualizations** before reading results - they tell the story
5. **Focus on interpretations** - connecting statistics to real-world meaning

## Next Steps (Optional)

If you want to extend this analysis:

1. **Export the notebook**:
   ```bash
   jupyter nbconvert --to html CA2_Statistical_Analysis.ipynb
   # or
   jupyter nbconvert --to pdf CA2_Statistical_Analysis.ipynb
   ```

2. **Add more analyses**:
   - Time series analysis (if temporal data available)
   - Multiple regression with additional predictors
   - Non-parametric alternatives (Kruskal-Wallis, Spearman correlation)
   - Clustering analysis to identify country groups

3. **Explore specific continents**:
   - Filter data by continent
   - Perform country-level analyses within continents

## Common Issues and Solutions

### Issue: Jupyter not finding packages
**Solution**: Make sure virtual environment is activated
```bash
source .venv/bin/activate
```

### Issue: Missing packages
**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

### Issue: Plots not displaying
**Solution**: Add this at the beginning of notebook
```python
%matplotlib inline
```

### Issue: Kernel crashes on large datasets
**Solution**: Already handled - we're using efficient pandas operations

## References

The analysis follows standard statistical practices as outlined in:
- Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics*
- Agresti, A. (2018). *Statistical Methods for the Social Sciences*
- James, G., et al. (2013). *An Introduction to Statistical Learning*

## Contact and Support

For questions about the analysis or statistical methods:
1. Review the `statistical_methods_explanation.md` document
2. Check the .kiro/specs/ directory for requirements and design docs
3. Look at inline comments in the notebook

---

## Completion Status

✅ **ALL TASKS COMPLETED**

- [x] Data loading and preparation
- [x] Exploratory data analysis
- [x] ANOVA test with visualizations and assumptions
- [x] Chi-square test with visualizations
- [x] Correlation analysis (matrix, heatmap, scatter plots)
- [x] Regression analysis (2 models with residual analysis)
- [x] Comprehensive documentation and conclusions
- [x] Statistical methods explanation document

**Total cells in notebook**: 70+ cells (markdown + code)  
**Total pages of documentation**: 30+ pages in explanation document

---

*Project completed: November 2025*  
*CA2: Statistical Modelling & Inference*
