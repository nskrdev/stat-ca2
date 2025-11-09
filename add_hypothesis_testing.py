import json

# Read the existing notebook
with open('CA2_Statistical_Analysis.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# New cells for hypothesis testing section
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "# Section 3: Hypothesis Testing\n",
            "\n",
            "In this section, we perform statistical hypothesis tests to answer key research questions about COVID-19 patterns across continents."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3.1 ANOVA Test: Death Rates Across Continents\n",
            "\n",
            "### Research Question\n",
            "**Is there a significant difference in death rates (Deaths/1M pop) across continents?**\n",
            "\n",
            "### Hypotheses\n",
            "- **Null Hypothesis (H₀)**: There is no significant difference in mean death rates across continents. All continent means are equal.\n",
            "- **Alternative Hypothesis (H₁)**: At least one continent has a significantly different mean death rate from the others.\n",
            "\n",
            "### Significance Level\n",
            "α = 0.05\n",
            "\n",
            "### Method\n",
            "We will use **one-way ANOVA** (Analysis of Variance) to compare death rates across multiple continents simultaneously."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Prepare data for ANOVA test\n",
            "# Filter out rows with missing death rates\n",
            "df_anova = df_analysis[df_analysis['Deaths/1M pop'].notna()].copy()\n",
            "\n",
            "print(\"ANOVA Test: Death Rates Across Continents\")\n",
            "print(\"=\"*80 + \"\\n\")\n",
            "\n",
            "print(f\"Sample size: {len(df_anova)} countries with complete death rate data\\n\")\n",
            "\n",
            "# Display sample sizes by continent\n",
            "print(\"Sample sizes by continent:\")\n",
            "continent_counts = df_anova['Continent'].value_counts().sort_index()\n",
            "for continent, count in continent_counts.items():\n",
            "    print(f\"  {continent}: {count} countries\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*80 + \"\\n\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Group death rates by continent\n",
            "continents = df_anova['Continent'].unique()\n",
            "death_rates_by_continent = []\n",
            "\n",
            "print(\"Descriptive Statistics by Continent:\\n\")\n",
            "print(f\"{'Continent':<20} {'Mean':<12} {'Median':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}\")\n",
            "print(\"-\" * 80)\n",
            "\n",
            "for continent in sorted(continents):\n",
            "    death_rates = df_anova[df_anova['Continent'] == continent]['Deaths/1M pop'].values\n",
            "    death_rates_by_continent.append(death_rates)\n",
            "    \n",
            "    print(f\"{continent:<20} {death_rates.mean():<12.2f} {np.median(death_rates):<12.2f} \"\n",
            "          f\"{death_rates.std():<12.2f} {death_rates.min():<12.2f} {death_rates.max():<12.2f}\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*80 + \"\\n\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Perform one-way ANOVA\n",
            "f_statistic, p_value = f_oneway(*death_rates_by_continent)\n",
            "\n",
            "print(\"ANOVA Test Results:\\n\")\n",
            "print(f\"F-statistic: {f_statistic:.4f}\")\n",
            "print(f\"P-value: {p_value:.6f}\")\n",
            "print(f\"Significance level (α): 0.05\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*80 + \"\\n\")\n",
            "\n",
            "# Interpret results\n",
            "print(\"Statistical Conclusion:\\n\")\n",
            "if p_value < 0.05:\n",
            "    print(f\"✓ REJECT the null hypothesis (p = {p_value:.6f} < 0.05)\")\n",
            "    print(\"\\nInterpretation:\")\n",
            "    print(\"There IS a statistically significant difference in death rates across continents.\")\n",
            "    print(\"At least one continent has a significantly different mean death rate from the others.\")\n",
            "else:\n",
            "    print(f\"✗ FAIL TO REJECT the null hypothesis (p = {p_value:.6f} >= 0.05)\")\n",
            "    print(\"\\nInterpretation:\")\n",
            "    print(\"There is NO statistically significant difference in death rates across continents.\")\n",
            "    print(\"The observed differences could be due to random chance.\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)"
        ]
    }
]

# Add the new cells to the notebook
notebook['cells'].extend(new_cells)

# Write the updated notebook
with open('CA2_Statistical_Analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Successfully added ANOVA test cells to notebook!")
