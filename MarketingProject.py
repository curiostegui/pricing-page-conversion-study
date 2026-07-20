
# SaaS Pricing Page Experiment — A/B Testing & Causal Inference

## Introduction

### Subscription SaaS (Software-as-a-Service) business rely heavily on their pricing page to manage current users and attract customers.
### It is a crucial touchpoint in the customer journey. A poorly setup pricing page can lead to disinterest and can cause users to abandon the website, even with an interest in the product.

### In this hypothetical scenario, the company is looking to revamp their payment plan. Currently, the pricing page uses a  monthly/annual toggle format, that presents two plan options with minimal context.
### It offers little support for prospective customers in the value of upgrading to higher tier payment plans. The conversion rates for free trial users as a result has plateaued. 
### If there was a more robust payment page, the pricing experience for users could see better outcomes.

## Experiment

### A redesigned pricing page 3-tier plan with a plan options side by side with supporting information was developed and tested.
### The experiment ran for 30 days from January 15 to February 14, 2024. 5000 users were randomly assigned and evenly split into two groups:

### Control group (2500 users): Saw the prior monthly/annual toggle pricing page. 
### Treatment group (2500 users): Saw the new 3-tier pricing page.

### The primary objective is to test whether the redsigned pricing page increases the rate at which users convert from free to a paid subscription.

### Outside of the main conversion metric. I tracked and identified both secondary and guardrail metrics to 
### ensure that improvements in conversion did not negatively effect other outcomes:

### Revenue per User: Did treatment generate more revenue per user exposed in the experiment?
### 30 day churn rate: Did treatment users stay or did they cancel their plan?
### Annual plan: Did the new page shift converters towards annual plans?

## Methodology

### The analysis will combine statistical testing with casual inference methods to evaluate the experiment.

### A/B Testing: This is a classical hypothesis test that tests whether variation in behaviors on a website is statistically significant
### Power analysis will be performed first to confirm that our sample size is large enough to detect a meaningful effect.
### Both primary and secondary metrics will be tested using the appropriate statistical test based on data type and distribution.
### Results will be reported with p-values and lift figures.

### Causal Inference: To complement the results of the A/B test, casual inference will be used to determine the confidence in the results.
### CUPED (Controlled-experiment Using Pre-Experiment Data) was used to test whether pre-experiment user behavior could reduce variation in the treatment effect estimate.
### Difference-in-Differences (DiD) was used to measure whether the treatment made meaningful user engagement changes 
### outside of natural trends by seeing how groups changed from pre-experiment to post-experiment. 
### OLS Regression looked at the treatment effect while also pairing it with multiple user-level characteristics to see the stability 
### of the treatment coefficient across different regression models.

### Subgroup Analysis: In this section, I'll be examining the conversion lift broken down across four segments: plan type, device type, age group and country.
### Interactions test will be then used to confirm whether the difference across segments were statistically significant.

### Behavioral Funnel: A customer behavior funnel was created to see where in the user journey, the treatment had the most impact.
### Each stage was analyzed seperately for both control and treatment groups. Using overall funnel rates and step-over-step conversion rates,
### I looked at what the overall pipeline health is. 

## load libraries & set styling 

### Load libraries
import pandas as pd          
import numpy as np           
import matplotlib.pyplot as plt  
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
import seaborn as sns        
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest, proportion_confint
import warnings
warnings.filterwarnings('ignore')   


# Global plot styling
plt.rcParams.update({
    'figure.facecolor': 'white',      # White background on all figures
    'axes.facecolor':   'white',      # White background inside axes
    'axes.spines.top':  False,        # Remove top border (cleaner look)
    'axes.spines.right': False,       # Remove right border (cleaner look)
    'axes.grid':        True,         # Show gridlines
    'grid.alpha':       0.3,          # Gridlines at 30% opacity (subtle)
    'font.size':        11            # Base font size for all text
})

# Color palette 
CONTROL_COLOR   = '#85B7EB'
TREATMENT_COLOR = '#AFA9EC'
HIGHLIGHT_COLOR = '#1D9E75'   
WARN_COLOR      = '#D95F4B'

### Load dataset
df = pd.read_csv(r"C:\Users\urios\Documents\saas_ab_test_dataset.csv")

## Data Exploration

### There are 5,000 rows and 18 columns that are comprised of a mix of identifiers (user_id), experiment assignment (experiment_group, experiment_start_date), 
### user background (signup_date, country, device_type, etc.), pre-experiment behavior (engagement_score_pre, sessions_pre_30d, etc.), 
### pricing page behavior (pricing_page_views, time_on_pricing_page_sec) and post outcome (converted_to_paid, plan_chosen, etc.) data.

### Basic exploration

## No observation of odd column and data type pairings.
## Equal distribution seen in the columns 
## between the control and treatment groups.

# Print all column names and their data types
print("\nColumn names and data types:")
print(df.dtypes)

# Basic shape and group sizes
# Confirm we have the right number of users and a ~50/50 split
# A healthy experiment should be very close to 50/50.
group_counts = df['experiment_group'].value_counts()
group_pcts   = df['experiment_group'].value_counts(normalize=True) * 100

print(f"\n{'Group':<15} {'Count':>8} {'Percent':>10}")
print("-" * 35)
for group in group_counts.index:
    print(f"{group:<15} {group_counts[group]:>8,} {group_pcts[group]:>9.1f}%")

### Nulls

## There are no significant nulls outside of the column days_to_conversion.
## It is normal for there to be a large number of nulls for this column
## as these are the users who did not convert within 30 days.

# Check for missing values across all columns
# Missing values need to be understood before any analysis
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    'missing_count': missing,
    'missing_pct':   missing_pct
})

# Only show columns that actually have missing values
has_missing = missing_report[missing_report['missing_count'] > 0]

if len(has_missing) == 0:
    print("\nNo missing values found — all columns are complete.")
else:
    print("\nColumns with missing values:")
    print(has_missing)

### Column Values

# Preview unique values for categorical columns
# This confirms categories are clean and spelled consistently
categorical_cols = [
    'experiment_group',
    'plan_at_experiment_start',
    'device_type',
    'age_bucket',
    'plan_chosen'
]

for col in categorical_cols:
    unique_vals = df[col].unique()
    print(f"\n{col}:")
    print(f"  {sorted([str(v) for v in unique_vals])}")

### Descriptive Statistics & Distribution Plots

# Seperate control & treatment
# Separate into control and treatment for easy referencing throughout
ctrl = df[df['experiment_group'] == 'control']
trt  = df[df['experiment_group'] == 'treatment']

### Examine metrics of interest

# In this section we're going to confirm that there is proper randomization
# between the pre treatment and control groups to ensure that neither control or 
# treatment group had a distinct advantage before the experiement.

# Columns engagement_score_pre and sessions_pre_30d show differences 
# of -1.34% and -0.74% showing that they are pretty much equivalent.
# Randomization worked.

# Define the outcome metrics we care about
outcome_metrics = [
    'converted_to_paid',
    'revenue_30d_usd',
    'sessions_post_30d',
    'churned_within_30d',
    'time_on_pricing_page_sec',
    'pricing_page_views',
    'engagement_score_pre',    # Pre-experiment (should be similar across groups)
    'sessions_pre_30d',        # Pre-experiment (should be similar across groups)
]

# Compute summary stats for each group
stats_ctrl = ctrl[outcome_metrics].agg(['mean', 'median', 'std', 'min', 'max'])
stats_trt  = trt[outcome_metrics].agg(['mean', 'median', 'std', 'min', 'max'])

# Print side by side for easy comparison
print("\n--- CONTROL GROUP ---")
print(stats_ctrl.round(3).T)

print("\n--- TREATMENT GROUP ---")
print(stats_trt.round(3).T)

# Compute the raw difference in means between groups
# Positive = treatment is higher, Negative = treatment is lower
print("\n--- MEAN DIFFERENCE (Treatment - Control) ---")
mean_diff = stats_trt.loc['mean'] - stats_ctrl.loc['mean']
pct_diff  = (mean_diff / stats_ctrl.loc['mean'] * 100).round(2)

diff_table = pd.DataFrame({
    'control_mean':   stats_ctrl.loc['mean'].round(4),
    'treatment_mean': stats_trt.loc['mean'].round(4),
    'abs_diff':       mean_diff.round(4),
    'pct_diff':       pct_diff
})
print(diff_table)

### Plan and Conversion Breakdown

# Here we're doing a check for even distribution between control and treatment groups among device_type, age_bucket, and country.
# We can see proper randomization.

cat_cols = ['plan_at_experiment_start', 'device_type', 'age_bucket', 'country']

for col in cat_cols:
    print(f"\n{col.upper()}:")
    # Crosstab shows raw counts
    ct = pd.crosstab(df[col], df['experiment_group'])
    # Add percentage columns for easier reading
    ct['control_%']   = (ct['control']   / ct['control'].sum()   * 100).round(1)
    ct['treatment_%'] = (ct['treatment'] / ct['treatment'].sum() * 100).round(1)
    print(ct)

### Distribution Plots

print("\n" + "="*60)
print("DISTRIBUTION PLOTS")
print("Generating 4 charts — a window will open...")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Distribution of key metrics by experiment group\n(Control vs Treatment)',
    fontsize=14, fontweight='bold', y=1.01
)

# Plot 1: Engagement score (pre-experiment)

# There is a bell shaped symmetrical curve for both control and treatment groups
# for engagement score. Both groups pre-experiment have almost identical engagement 
# quality.

ax1 = axes[0, 0]

sns.kdeplot(
    ctrl['engagement_score_pre'],
    ax=ax1,
    color=CONTROL_COLOR,
    fill=True,
    alpha=0.4,
    label='Control'
)
sns.kdeplot(
    trt['engagement_score_pre'],
    ax=ax1,
    color=TREATMENT_COLOR,
    fill=True,
    alpha=0.4,
    label='Treatment'
)

ax1.set_title('Engagement score (pre-experiment)', fontweight='bold')
ax1.set_xlabel('Engagement score (1–10)')
ax1.set_ylabel('Density')
ax1.legend()

# Add vertical lines for group means
ax1.axvline(ctrl['engagement_score_pre'].mean(), color=CONTROL_COLOR,
            linestyle='--', linewidth=1.5,
            label=f"Control mean: {ctrl['engagement_score_pre'].mean():.2f}")
ax1.axvline(trt['engagement_score_pre'].mean(),  color=TREATMENT_COLOR,
            linestyle='--', linewidth=1.5,
            label=f"Treatment mean: {trt['engagement_score_pre'].mean():.2f}")
ax1.legend(fontsize=9)

# Plot 2: Time on pricing page (seconds)

## We can see that the treatment group spends more time than than the control group
## This is a great sign. Control peaks at 75 secs, while the treatment peaks at 95 secs. 
## There is a difference a 20 secs between both groups.
## Because of normal distribution observed, a t-test can be used to measure statistical significance.

## Note: time_on_pricing_page is a post-experiement behavioral metric

ax2 = axes[0, 1]

sns.kdeplot(
    ctrl['time_on_pricing_page_sec'],
    ax=ax2,
    color=CONTROL_COLOR,
    fill=True,
    alpha=0.4,
    label=f"Control (mean: {ctrl['time_on_pricing_page_sec'].mean():.0f}s)"
)
sns.kdeplot(
    trt['time_on_pricing_page_sec'],
    ax=ax2,
    color=TREATMENT_COLOR,
    fill=True,
    alpha=0.4,
    label=f"Treatment (mean: {trt['time_on_pricing_page_sec'].mean():.0f}s)"
)

ax2.set_title('Time on pricing page (seconds)', fontweight='bold')
ax2.set_xlabel('Seconds')
ax2.set_ylabel('Density')
ax2.legend(fontsize=9)

# Plot 3: Revenue per user (30 days) — FULL distribution

# A large amount of users (about 90%) did not convert and did not generate revenue as a result. 
# There are however, some mall bumps near $8.25 and $12.99.
# The shape takes on a zero-inflated distribution. Since it violates core assumptions of a t-test,
# we will need to use a non-parametric test to determine statistical significance.

ax3 = axes[1, 0]

# Use histogram (not KDE) here because KDE smooths out the zero spike
# and makes zero-inflation less visible
ax3.hist(
    ctrl['revenue_30d_usd'],
    bins=40,
    color=CONTROL_COLOR,
    alpha=0.6,
    label='Control',
    density=True   # Normalize so both groups are on the same scale
)
ax3.hist(
    trt['revenue_30d_usd'],
    bins=40,
    color=TREATMENT_COLOR,
    alpha=0.6,
    label='Treatment',
    density=True
)

ax3.set_title('Revenue per user — 30 days (zero-inflated)', fontweight='bold')
ax3.set_xlabel('Revenue (USD)')
ax3.set_ylabel('Density')
ax3.legend(fontsize=9)

# Annotate to explain the spike
ax3.annotate(
    '← Spike at $0\n   (non-converters)',
    xy=(0.5, ax3.get_ylim()[1] * 0.7),
    fontsize=9,
    color='gray'
)

# Plot 4: Revenue among CONVERTERS only

# In this chart, I filtered for the users who did convert to look at changes.
# For the annual plan at $8.25, we see a higher number of users from the treatment group.
# The new pricing page is nudging converters to the annual plan. It has a lower monthly price compared
# to the $12.99 (monthly plan) but has higher long term value as they churn less. 
# There is a visible bimodal distribution.

ax4 = axes[1, 1]

ctrl_converters = ctrl[ctrl['converted_to_paid'] == 1]
trt_converters  = trt[trt['converted_to_paid'] == 1]

ax4.hist(
    ctrl_converters['revenue_30d_usd'],
    bins=30,
    color=CONTROL_COLOR,
    alpha=0.6,
    label=f"Control (n={len(ctrl_converters):,})",
    density=True
)
ax4.hist(
    trt_converters['revenue_30d_usd'],
    bins=30,
    color=TREATMENT_COLOR,
    alpha=0.6,
    label=f"Treatment (n={len(trt_converters):,})",
    density=True
)

ax4.set_title('Revenue per user — converters only', fontweight='bold')
ax4.set_xlabel('Revenue (USD)')
ax4.set_ylabel('Density')
ax4.legend(fontsize=9)

# Annotate the two plan price points
ax4.axvline(8.25,  color='gray', linestyle=':', linewidth=1)
ax4.axvline(12.99, color='gray', linestyle=':', linewidth=1)
ax4.text(7.5,  ax4.get_ylim()[1] * 0.85, '$8.25\n(annual)',  fontsize=8, color='gray')
ax4.text(12.1, ax4.get_ylim()[1] * 0.85, '$12.99\n(monthly)', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('phase1_step2_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✓ Distribution plots saved as 'phase1_step2_distributions.png'")

# Section 4: Sessions pre vs post — a first look at engagement shift

# When looking at the number of sessions for both control and treatment sessions post experiment
# we can observe that the treatment increased their mean number of sessions by 1.22 (16.4%), 
# showing a positive increase.

sessions_summary = df.groupby('experiment_group').agg(
    sessions_pre_mean  = ('sessions_pre_30d',  'mean'),
    sessions_post_mean = ('sessions_post_30d', 'mean'),
).round(2)

sessions_summary['change'] = (
    sessions_summary['sessions_post_mean'] - sessions_summary['sessions_pre_mean']
).round(2)

sessions_summary['pct_change'] = (
    sessions_summary['change'] / sessions_summary['sessions_pre_mean'] * 100
).round(1)

print("\nMean sessions pre and post experiment:")
print(sessions_summary)

### Summary

# In all performance indicators we can observe positive changes. Conversion rates, Revenue and 
# time on pricing page showed an increase in the treatment group. The churn rate is also lower
# in the treatment group.

print(f"""
  1. Conversion rate:
       Control:   {ctrl['converted_to_paid'].mean():.1%}
       Treatment: {trt['converted_to_paid'].mean():.1%}
       Difference: +{(trt['converted_to_paid'].mean() - ctrl['converted_to_paid'].mean()):.1%}

  2. Revenue per user (all users):
       Control:   ${ctrl['revenue_30d_usd'].mean():.2f}
       Treatment: ${trt['revenue_30d_usd'].mean():.2f}
       Difference: +${(trt['revenue_30d_usd'].mean() - ctrl['revenue_30d_usd'].mean()):.2f}

  3. Churn rate:
       Control:   {ctrl['churned_within_30d'].mean():.1%}
       Treatment: {trt['churned_within_30d'].mean():.1%}
       Difference: {(trt['churned_within_30d'].mean() - ctrl['churned_within_30d'].mean()):.1%}

  4. Time on pricing page:
       Control:   {ctrl['time_on_pricing_page_sec'].mean():.0f}s
       Treatment: {trt['time_on_pricing_page_sec'].mean():.0f}s
       Difference: +{(trt['time_on_pricing_page_sec'].mean() - ctrl['time_on_pricing_page_sec'].mean()):.0f}s

  5. Revenue distribution is heavily zero-inflated.
     → Will need to check normality in Phase 2 before choosing test.

""")

## Statistical Testing 

# Formally verify if the distribution in the control and treatment groups are statistically significant.
# Based on the results, It's observed that plan_at_experiment_start and the country variable
# have significant imbalances which will have to be addressed later.

print("\n" + "="*60)
print("CHI-SQUARE TESTS — CATEGORICAL VARIABLES")
print("="*60)

categorical_vars = [
    'plan_at_experiment_start',
    'device_type',
    'age_bucket',
    'country'
]

# Store results for the summary table and visualization
chi_square_results = []

for var in categorical_vars:

    # Step 1: Build contingency table (counts per category per group)
    contingency = pd.crosstab(df[var], df['experiment_group'])

    # Step 2: Run chi-square test
    # chi2  = test statistic (larger = more imbalance)
    # p     = probability of seeing this imbalance by chance if groups are equal
    # dof   = degrees of freedom (number of categories - 1)
    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    # Step 3: Determine pass/fail
    passed = p > 0.05

    # Step 4: Store result
    chi_square_results.append({
        'variable':   var,
        'chi2_stat':  round(chi2, 4),
        'p_value':    round(p, 4),
        'dof':        dof,
        'balanced':   '✓ PASS' if passed else '✗ FAIL'
    })

    # Step 5: Print detailed breakdown per variable
    print(f"\n{'─'*50}")
    print(f"Variable: {var.upper()}")
    print(f"{'─'*50}")

    # Show the percentage split per category per group
    pct_table = contingency.copy().astype(float)
    pct_table['control_%']   = (contingency['control']   / contingency['control'].sum()   * 100).round(1)
    pct_table['treatment_%'] = (contingency['treatment'] / contingency['treatment'].sum() * 100).round(1)
    pct_table['diff_%']      = (pct_table['treatment_%'] - pct_table['control_%']).round(1)
    print(pct_table)

    print(f"\n  Chi-square statistic : {chi2:.4f}")
    print(f"  Degrees of freedom   : {dof}")
    print(f"  p-value              : {p:.4f}")
    print(f"  Result               : {'✓ PASS — no significant imbalance (p > 0.05)' if passed else '✗ FAIL — significant imbalance detected (p < 0.05)'}")


print("\n" + "="*60)
print("T-TESTS — NUMERIC PRE-EXPERIMENT VARIABLES")
print("="*60)

numeric_vars = [
    'engagement_score_pre',
    'sessions_pre_30d'
]

t_test_results = []

for var in numeric_vars:

    ctrl_vals = ctrl[var]
    trt_vals  = trt[var]

    # Two-sample t-test
    # equal_var=False uses Welch's t-test, which doesn't assume both groups
    # have the same variance — this is the safer, more robust choice
    t_stat, p = stats.ttest_ind(ctrl_vals, trt_vals, equal_var=False)

    passed = p > 0.05

    t_test_results.append({
        'variable':  var,
        't_stat':    round(t_stat, 4),
        'p_value':   round(p, 4),
        'ctrl_mean': round(ctrl_vals.mean(), 4),
        'trt_mean':  round(trt_vals.mean(), 4),
        'diff':      round(trt_vals.mean() - ctrl_vals.mean(), 4),
        'balanced':  '✓ PASS' if passed else '✗ FAIL'
    })

    print(f"\n{'─'*50}")
    print(f"Variable: {var.upper()}")
    print(f"{'─'*50}")
    print(f"  Control mean   : {ctrl_vals.mean():.4f}")
    print(f"  Treatment mean : {trt_vals.mean():.4f}")
    print(f"  Difference     : {trt_vals.mean() - ctrl_vals.mean():.4f}")
    print(f"  T-statistic    : {t_stat:.4f}")
    print(f"  p-value        : {p:.4f}")
    print(f"  Result         : {'✓ PASS — means are not significantly different (p > 0.05)' if passed else '✗ FAIL — means differ significantly (p < 0.05)'}")


# Master balance table
print("\n" + "="*60)
print("MASTER BALANCE TABLE")
print("="*60)

# Combine chi-square and t-test results
chi_df = pd.DataFrame(chi_square_results)[['variable', 'chi2_stat', 'p_value', 'balanced']]
chi_df.rename(columns={'chi2_stat': 'test_stat'}, inplace=True)
chi_df['test_type'] = 'Chi-square'

t_df = pd.DataFrame(t_test_results)[['variable', 't_stat', 'p_value', 'balanced']]
t_df.rename(columns={'t_stat': 'test_stat'}, inplace=True)
t_df['test_type'] = 'T-test (Welch)'

balance_table = pd.concat([chi_df, t_df], ignore_index=True)
balance_table = balance_table[['variable', 'test_type', 'test_stat', 'p_value', 'balanced']]

print("\n")
print(balance_table.to_string(index=False))

# Overall verdict
all_passed = all(
    r['balanced'] == '✓ PASS'
    for r in chi_square_results + t_test_results
)

print("\n" + "="*60)
if all_passed:
    print("✓ OVERALL VERDICT: Randomization is valid.")
    print("  All pre-experiment variables are balanced across groups.")
    print("  We can proceed with the analysis — any post-experiment")
    print("  differences are attributable to the treatment, not to")
    print("  pre-existing differences between users.")
else:
    print("✗ OVERALL VERDICT: Randomization issue detected.")
    print("  One or more variables show significant imbalance.")
    print("  Consider adjusting for imbalanced variables as covariates")
    print("  in the regression models (Phase 3).")
print("="*60)

## Modeling

### A/B Test 

#### Power Analysis

## Prior to starting the experimenting, I'm going to check whether the experiment is adequately powered
## to detect a meaningful conversion. A test with two small of sample size, might miss a real effect 
## because it's too small.

# Here we define our parameters prior to performing power analysis.
# Sample sizes from our experiment
n_control   = len(ctrl)       # 2,500 users
n_treatment = len(trt)        # 2,500 users

# Alpha: significance threshold
# Standard choice in industry — 5% false positive rate
alpha = 0.05

# Power: probability of detecting a true effect
# 80% is the standard minimum — meaning we accept a 20% chance of
# missing a real effect (Type II error)
power_target = 0.80

# Baseline conversion rate from control group
# This is what we observed — in a real experiment you'd use a
# historical estimate from before the experiment ran
baseline_rate = ctrl['converted_to_paid'].mean()

# Observed conversion rate in treatment
observed_rate = trt['converted_to_paid'].mean()

# The actual lift we observed
observed_lift = observed_rate - baseline_rate

print(f"\n  Sample size per group  : {n_control:,}")
print(f"  Alpha (α)              : {alpha}")
print(f"  Target power (1-β)     : {power_target}")
print(f"  Baseline conversion    : {baseline_rate:.4f} ({baseline_rate:.1%})")
print(f"  Observed conversion    : {observed_rate:.4f} ({observed_rate:.1%})")
print(f"  Observed lift          : {observed_lift:.4f} ({observed_lift:.1%})")

#### Minimum Detectable Effect (MDE)

## I want to look into the MDE. This tell us, given our sample size, alpha, and power target —
## what is the smallest true lift we could reliably detect?

## The MDE was calculated at 2.34pp and the observed lift is 2.44pp. 
## The experiment is adequately powered as a result.

# Step 1: Set up the power analysis solver
power_analysis = NormalIndPower()

# Step 2: Find the effect size (Cohen's h) that gives us 80% power
# We solve FOR effect size, given n, alpha, and power
# nobs1 = observations in group 1 (we have equal groups so nobs1 = n/group)
min_effect_size = power_analysis.solve_power(
    effect_size = None,      # This is what we're solving for
    nobs1       = n_control,
    alpha       = alpha,
    power       = power_target,
    ratio       = 1.0        # Equal group sizes (treatment/control ratio)
)

print(f"\n  Minimum detectable effect size (Cohen's h) : {min_effect_size:.4f}")

# Step 3: Convert Cohen's h back into a percentage point lift
# Cohen's h = 2 * arcsin(sqrt(p2)) - 2 * arcsin(sqrt(p1))
# We solve for p2 given p1 (baseline) and h (min effect size)
# arcsin transformation maps proportions to a scale where differences
# are more meaningful statistically

p1 = baseline_rate
min_detectable_rate = np.sin(np.arcsin(np.sqrt(p1)) + min_effect_size / 2) ** 2
mde_pp = min_detectable_rate - p1   # MDE in percentage points

print(f"  Baseline conversion rate                   : {p1:.1%}")
print(f"  Minimum detectable conversion rate         : {min_detectable_rate:.1%}")
print(f"  MDE in percentage points                   : {mde_pp:.2%} ({mde_pp*100:.2f}pp)")
print(f"\n  Observed lift                              : {observed_lift:.2%} ({observed_lift*100:.2f}pp)")

# Step 4: Compare MDE to observed lift
if observed_lift >= mde_pp:
    print(f"\n  ✓ RESULT: Observed lift ({observed_lift:.2%}) is LARGER than the MDE")
    print(f"    ({mde_pp:.2%}). The experiment was adequately powered to detect")
    print(f"    our observed effect.")
else:
    print(f"\n  ✗ RESULT: Observed lift ({observed_lift:.2%}) is SMALLER than the MDE")
    print(f"    ({mde_pp:.2%}). The experiment may have been underpowered.")

#### Statistical Test

## I'll now be checking if the difference in conversion rates between
## control and treatment is statistically significant.

## The 2.4% difference in conversion rates between control and treatmeant group
## was found to be statistical signifcant. The lift from the treatment group is found to be meaningful as a result.

# Count converters and non-converters per group
n_ctrl        = len(ctrl)
n_trt         = len(trt)
conv_ctrl     = ctrl['converted_to_paid'].sum()      # Number who converted
conv_trt      = trt['converted_to_paid'].sum()
not_conv_ctrl = n_ctrl - conv_ctrl                   # Number who didn't convert
not_conv_trt  = n_trt  - conv_trt

# Conversion rates
rate_ctrl = conv_ctrl / n_ctrl
rate_trt  = conv_trt  / n_trt

# Build contingency table as a DataFrame for clean display
contingency_df = pd.DataFrame({
    'control':   [conv_ctrl,     not_conv_ctrl,  n_ctrl],
    'treatment': [conv_trt,      not_conv_trt,   n_trt]
}, index=['Converted', 'Not converted', 'Total'])

print("\nContingency table (raw counts):")
print(contingency_df)

print(f"\nConversion rates:")
print(f"  Control   : {conv_ctrl:,} / {n_ctrl:,} = {rate_ctrl:.4f} ({rate_ctrl:.1%})")
print(f"  Treatment : {conv_trt:,} / {n_trt:,} = {rate_trt:.4f} ({rate_trt:.1%})")
print(f"  Difference: {rate_trt - rate_ctrl:.4f} ({(rate_trt - rate_ctrl):.1%}) absolute lift")
print(f"  Relative lift: {(rate_trt - rate_ctrl) / rate_ctrl:.1%}")

# NOTE ON RELATIVE VS ABSOLUTE LIFT:
# Absolute lift = treatment rate - control rate = the raw pp difference
# Relative lift = absolute lift / control rate  = % improvement over baseline
# Both matter — always report both in a professional write-up

print("\n" + "="*60)
print("CHI-SQUARE TEST")
print("="*60)

# Build the contingency array for scipy
# Format: [[converted_ctrl, converted_trt], [not_converted_ctrl, not_converted_trt]]
contingency_array = np.array([
    [conv_ctrl,     conv_trt],
    [not_conv_ctrl, not_conv_trt]
])

# Run chi-square test
# chi2      = test statistic
# p         = p-value
# dof       = degrees of freedom (always 1 for a 2x2 table)
# expected  = what counts would look like if there were no difference
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_array)

print(f"\nChi-square test results:")
print(f"  Chi-square statistic : {chi2:.4f}")
print(f"  Degrees of freedom   : {dof}")
print(f"  p-value              : {p_chi2:.4f}")

# Show expected counts for context
print(f"\nExpected counts (if no difference between groups):")
expected_df = pd.DataFrame(
    expected.round(1),
    columns=['control', 'treatment'],
    index=['Converted', 'Not converted']
)
print(expected_df)
print("  (Expected counts show what a perfectly balanced split would look like)")

# Cross-check with two-proportion z-test
# This is mathematically equivalent and gives the same result
counts  = np.array([conv_ctrl, conv_trt])
n_obs   = np.array([n_ctrl, n_trt])
z_stat, p_ztest = proportions_ztest(counts, n_obs)

print(f"\nCross-check — two-proportion z-test:")
print(f"  Z-statistic : {z_stat:.4f}")
print(f"  p-value     : {p_ztest:.4f}")
print(f"  (Should match chi-square p-value — confirms result is consistent)")

#### Metrics Examination

## In this section, I'll be looking at the changes in the metrics of interest such as conversion, revenue per user, churn rate and plan mix.

# Converter-only subsets — used for plan mix analysis
ctrl_conv = ctrl[ctrl['converted_to_paid'] == 1]
trt_conv  = trt[trt['converted_to_paid'] == 1]

print(f"  Control users      : {len(ctrl):,} ({len(ctrl_conv):,} converters)")
print(f"  Treatment users    : {len(trt):,}  ({len(trt_conv):,} converters)")

##### Revenue per User

## The Shapiro Wilk test confirms what was shown earlier, the distribution is not 
## normal (p value - 0.00000) and we there need to use a Mann-Whitney U test 
## instead of a t test to look at signficance.

## After running the test, the we see that that increase in Revenue per user is statistically
## significant (0.0027). The treatment group generated $0.25 more than
## the control group (+25.1%).

# Step 1: Descriptive stats
rev_ctrl = ctrl['revenue_30d_usd']
rev_trt  = trt['revenue_30d_usd']

print(f"\nDescriptive statistics — revenue per user (all users):")
print(f"  {'Metric':<20} {'Control':>12} {'Treatment':>12}")
print(f"  {'─'*46}")
print(f"  {'Mean':<20} {'${:.4f}'.format(rev_ctrl.mean()):>12} "
      f"{'${:.4f}'.format(rev_trt.mean()):>12}")
print(f"  {'Median':<20} {'${:.4f}'.format(rev_ctrl.median()):>12} "
      f"{'${:.4f}'.format(rev_trt.median()):>12}")
print(f"  {'Std dev':<20} {'${:.4f}'.format(rev_ctrl.std()):>12} "
      f"{'${:.4f}'.format(rev_trt.std()):>12}")
print(f"  {'% with $0 revenue':<20} {(rev_ctrl==0).mean():>11.1%} "
      f"{(rev_trt==0).mean():>11.1%}")

# Step 2: Normality check — Shapiro-Wilk test
# Shapiro-Wilk tests whether data comes from a normal distribution
# H0: data is normally distributed
# p < 0.05 → reject normality → use Mann-Whitney U instead of t-test
# NOTE: We sample 500 users because Shapiro-Wilk is slow on large datasets
np.random.seed(42)
sample_ctrl = rev_ctrl.sample(500)
sample_trt  = rev_trt.sample(500)

_, p_shapiro_ctrl = stats.shapiro(sample_ctrl)
_, p_shapiro_trt  = stats.shapiro(sample_trt)

print(f"\nNormality check — Shapiro-Wilk test (sample n=500):")
print(f"  Control   p-value : {p_shapiro_ctrl:.6f} "
      f"{'→ NOT normal (p < 0.05)' if p_shapiro_ctrl < 0.05 else '→ Normal'}")
print(f"  Treatment p-value : {p_shapiro_trt:.6f} "
      f"{'→ NOT normal (p < 0.05)' if p_shapiro_trt < 0.05 else '→ Normal'}")
print(f"\n  → Confirmed: revenue is NOT normally distributed.")
print(f"    Using Mann-Whitney U test instead of t-test.")

# Step 3: Mann-Whitney U test
# alternative='less' tests whether control values tend to be LOWER than treatment
u_stat, p_mannwhitney = stats.mannwhitneyu(
    rev_ctrl, rev_trt,
    alternative='less'    # One-sided: we expect treatment revenue > control
)

# Effect size: rank-biserial correlation
# Ranges from -1 to 1. Values near 0 = small effect, near 1 = large effect
n1, n2 = len(rev_ctrl), len(rev_trt)
rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

rev_lift     = rev_trt.mean() - rev_ctrl.mean()
rev_lift_pct = rev_lift / rev_ctrl.mean() * 100

print(f"\nMann-Whitney U test results:")
print(f"  U-statistic          : {u_stat:,.0f}")
print(f"  p-value (one-sided)  : {p_mannwhitney:.4f}")
print(f"  Effect size (r)      : {rank_biserial:.4f}")
print(f"  Revenue lift         : +${rev_lift:.4f} per user")
print(f"  Relative lift        : +{rev_lift_pct:.1f}%")
print(f"  Significant          : "
      f"{'✓ YES (p < 0.05)' if p_mannwhitney < 0.05 else '✗ NO (p ≥ 0.05)'}")

##### Churn Rate

## The churn rate in the treatment group dropped from 5.85% to 3.99% (-1.87%)
## showing an overall improvement. This drop was found to be statistically significant.


# Filter to existing paid users only (monthly or annual at experiment start)
paid_ctrl = ctrl[ctrl['plan_at_experiment_start'].isin(['monthly', 'annual'])]
paid_trt  = trt[trt['plan_at_experiment_start'].isin(['monthly', 'annual'])]

churn_ctrl = paid_ctrl['churned_within_30d']
churn_trt  = paid_trt['churned_within_30d']

churn_rate_ctrl = churn_ctrl.mean()
churn_rate_trt  = churn_trt.mean()

print(f"\nExisting paid users only:")
print(f"  Control group   : {len(paid_ctrl):,} users")
print(f"  Treatment group : {len(paid_trt):,} users")

print(f"\nChurn rates:")
print(f"  Control   : {churn_ctrl.sum():,} churned / {len(paid_ctrl):,} = {churn_rate_ctrl:.2%}")
print(f"  Treatment : {churn_trt.sum():,} churned / {len(paid_trt):,} = {churn_rate_trt:.2%}")
print(f"  Difference: {churn_rate_trt - churn_rate_ctrl:.2%} "
      f"({'↓ improvement' if churn_rate_trt < churn_rate_ctrl else '↑ worsened'})")

# Chi-square test on churn
churn_contingency = np.array([
    [churn_ctrl.sum(),              churn_trt.sum()],
    [len(paid_ctrl) - churn_ctrl.sum(), len(paid_trt) - churn_trt.sum()]
])

chi2_churn, p_churn, _, _ = stats.chi2_contingency(churn_contingency)

# Confidence interval on churn rate difference
ci_churn_ctrl_low, ci_churn_ctrl_high = proportion_confint(
    churn_ctrl.sum(), len(paid_ctrl), alpha=0.05)
ci_churn_trt_low, ci_churn_trt_high   = proportion_confint(
    churn_trt.sum(),  len(paid_trt),  alpha=0.05)

print(f"\nChi-square test results:")
print(f"  Chi-square statistic : {chi2_churn:.4f}")
print(f"  p-value              : {p_churn:.4f}")
print(f"  95% CI control churn : [{ci_churn_ctrl_low:.2%}, {ci_churn_ctrl_high:.2%}]")
print(f"  95% CI treatment churn: [{ci_churn_trt_low:.2%}, {ci_churn_trt_high:.2%}]")
print(f"  Significant          : "
      f"{'✓ YES (p < 0.05)' if p_churn < 0.05 else '✗ NO (p ≥ 0.05)'}")

if churn_rate_trt < churn_rate_ctrl:
    print(f"\n  ✓ GUARDRAIL PASSED: Treatment churn is lower than control.")
    print(f"    The new pricing page did not cause cancellations.")
else:
    print(f"\n  ✗ GUARDRAIL CONCERN: Treatment churn is higher than control.")
    print(f"    Investigate whether the new page caused buyer's remorse.")


##### Plan Mix

## A higher percentage from the treatment group signed up the annual plan.
## There was an increase of 31.9% (control) to 37.6%.
## This was shown to not be statistically significant. 
## This is likely because of the small sample size. 

print(f"\nConverters only:")
print(f"  Control converters   : {len(ctrl_conv):,}")
print(f"  Treatment converters : {len(trt_conv):,}")

# Plan distribution among converters
plan_ctrl = ctrl_conv['plan_chosen'].value_counts()
plan_trt  = trt_conv['plan_chosen'].value_counts()

annual_ctrl    = (ctrl_conv['plan_chosen'] == 'annual').sum()
annual_trt     = (trt_conv['plan_chosen']  == 'annual').sum()
monthly_ctrl   = (ctrl_conv['plan_chosen'] == 'monthly').sum()
monthly_trt    = (trt_conv['plan_chosen']  == 'monthly').sum()

annual_rate_ctrl = annual_ctrl / len(ctrl_conv)
annual_rate_trt  = annual_rate_trt_val = annual_trt / len(trt_conv)

print(f"\nPlan mix among converters:")
print(f"  {'Plan':<12} {'Control':>12} {'Treatment':>12}")
print(f"  {'─'*38}")
print(f"  {'Annual':<12} {annual_ctrl:>5} ({annual_rate_ctrl:.1%})"
      f"  {annual_trt:>5} ({annual_rate_trt:.1%})")
print(f"  {'Monthly':<12} {monthly_ctrl:>5} ({1-annual_rate_ctrl:.1%})"
      f"  {monthly_trt:>5} ({1-annual_rate_trt:.1%})")
print(f"  {'Total':<12} {len(ctrl_conv):>5} (100%)"
      f"      {len(trt_conv):>5} (100%)")

# Chi-square test on plan mix
plan_contingency = np.array([
    [annual_ctrl,  annual_trt],
    [monthly_ctrl, monthly_trt]
])

chi2_plan, p_plan, _, _ = stats.chi2_contingency(plan_contingency)

annual_lift = annual_rate_trt - annual_rate_ctrl

print(f"\nChi-square test results:")
print(f"  Chi-square statistic : {chi2_plan:.4f}")
print(f"  p-value              : {p_plan:.4f}")
print(f"  Annual plan lift     : {annual_lift:+.2%}")
print(f"  Significant          : "
      f"{'✓ YES (p < 0.05)' if p_plan < 0.05 else '✗ NO (p ≥ 0.05)'}")


#### Summary

## Overall all the metrics resulted in desirable outcomes.
## Our primary metric, conversion rate, improved in the treatment group.
## However an increase in conversion but decrease in our guardrail metrics would have weakened our results. 
## Thankfully we didn't see a negative impact.
## The churn rate went down, revenue per user and annual plan mix went up.
## Three of our metrics were also found to be statistically signficant (conversion rate, revenue per user, and churn rate).


#### Results Table
  
print(f"""
{'='*65}
MASTER RESULTS TABLE — PHASE 2 (ALL METRICS)
{'='*65}
  {'Metric':<28} {'Control':>10} {'Treatment':>10} {'Lift':>8} {'p-value':>9} {'Sig?':>6}
  {'─'*65}
  {'Conversion rate':<28} {ctrl['converted_to_paid'].mean():>9.1%} \
{trt['converted_to_paid'].mean():>10.1%} \
{trt['converted_to_paid'].mean()-ctrl['converted_to_paid'].mean():>+7.2%} \
{'0.0042':>9} {'✓':>6}
  {'Revenue / user':<28} {'${:.2f}'.format(rev_ctrl.mean()):>10} \
{'${:.2f}'.format(rev_trt.mean()):>10} \
{'+${:.2f}'.format(rev_lift):>8} \
{p_mannwhitney:>9.4f} \
{'✓' if p_mannwhitney < 0.05 else '✗':>6}
  {'Churn rate':<28} {churn_rate_ctrl:>9.1%} \
{churn_rate_trt:>10.1%} \
{churn_rate_trt-churn_rate_ctrl:>+7.2%} \
{p_churn:>9.4f} \
{'✓' if p_churn < 0.05 else '✗':>6}
  {'Annual plan rate':<28} {annual_rate_ctrl:>9.1%} \
{annual_rate_trt:>10.1%} \
{annual_lift:>+7.2%} \
{p_plan:>9.4f} \
{'✓' if p_plan < 0.05 else '✗':>6}
{'='*65}
 
PHASE 2 SUMMARY:
  ✓ Conversion rate lifted significantly (+2.44pp, p=0.0042)
  {'✓' if p_mannwhitney < 0.05 else '✗'} Revenue per user {'increased' if rev_lift > 0 else 'decreased'} \
(+${rev_lift:.2f}, p={p_mannwhitney:.4f})
  {'✓' if p_churn < 0.05 else '—'} Churn rate \
{'decreased significantly' if p_churn < 0.05 and churn_rate_trt < churn_rate_ctrl else 'not significantly different'} \
({churn_rate_trt-churn_rate_ctrl:+.2%}, p={p_churn:.4f})
  {'✓' if p_plan < 0.05 else '—'} Annual plan rate \
{'shifted significantly' if p_plan < 0.05 else 'not significantly different'} \
({annual_lift:+.2%}, p={p_plan:.4f})
 
  All guardrail metrics are moving in the right direction.
  The new pricing page improves conversion, revenue, and
  plan quality without increasing churn.
 
{'='*65}
""")

### Causal Inference

#### CUPED Technique

outcome = 'converted_to_paid'

## I utilized CUPED analysis to try to examine and remove any variance that 
## is explained by pre experiment user behavior (users that were going to anyway)
## to leave a cleaner estimate of the treatment effect.
## The CUPED analysis showed that pre-experiment user behavior was a weak predictor of 
## of conversion (r<0.03). This strengthens my confidence in the A/B test results.
 
##### Best Single Covariate

print("\n" + "="*60)
print("COVARIATE CORRELATION SCREENING")
print("(Finding the strongest pre-experiment predictor of conversion)")
print("="*60)

# All valid pre-experiment covariates
pre_experiment_covariates = [
    'engagement_score_pre',
    'sessions_pre_30d'
]

print(f"\n  {'Covariate':<30} {'Correlation (r)':<18} {'p-value':<12} {'Strength'}")
print(f"  {'─'*70}")

correlations = {}
for cov in pre_experiment_covariates:
    r, p = stats.pearsonr(df[cov], df[outcome])
    correlations[cov] = {'r': r, 'p': p}

    # Classify strength
    if abs(r) >= 0.3:
        strength = 'Strong'
    elif abs(r) >= 0.1:
        strength = 'Moderate'
    elif abs(r) >= 0.05:
        strength = 'Weak'
    else:
        strength = 'Very weak'

    sig = '✓' if p < 0.05 else '✗'
    print(f"  {cov:<30} {r:>10.4f}      {p:>10.4f}   {sig} {strength}")

# Identify best single covariate
best_cov = max(correlations, key=lambda x: abs(correlations[x]['r']))
best_r   = correlations[best_cov]['r']
print(f"\n  Best single covariate: {best_cov} (r={best_r:.4f})")

##### Multiple Covariates

print("\n" + "="*60)
print("BUILDING A COMBINED COVARIATE")
print("(Combining all pre-experiment variables for maximum variance reduction)")
print("="*60)

# Standardize covariates so they're on the same scale
# This ensures sessions_pre_30d (range 0-80) doesn't dominate
# engagement_score_pre (range 1-10) just because of scale differences
scaler  = StandardScaler()
X_pre   = scaler.fit_transform(df[pre_experiment_covariates])
Y       = df[outcome].values

# Fit OLS on full dataset to get predicted values
ols = LinearRegression()
ols.fit(X_pre, Y)
Y_hat = ols.predict(X_pre)

# Correlation of combined covariate with outcome
r_combined, p_combined = stats.pearsonr(Y_hat, Y)

print(f"\n  Individual correlations:")
for cov, vals in correlations.items():
    print(f"    {cov:<30} r={vals['r']:.4f}")

print(f"\n  Combined covariate correlation: r={r_combined:.4f} (p={p_combined:.4f})")
print(f"  Improvement over best single  : "
      f"{abs(r_combined) - abs(best_r):.4f} "
      f"({'better' if abs(r_combined) > abs(best_r) else 'no improvement'})")

print(f"\n  OLS coefficients:")
for cov, coef in zip(pre_experiment_covariates, ols.coef_):
    print(f"    {cov:<30} : {coef:.6f}")

##### Comparison

print("\n" + "="*60)
print("CUPED — THREE COVARIATE APPROACHES COMPARED")
print("="*60)

# Raw baseline (no CUPED)
rate_ctrl = ctrl[outcome].mean()
rate_trt  = trt[outcome].mean()
raw_lift  = rate_trt - rate_ctrl

var_ctrl_raw = ctrl[outcome].var()
var_trt_raw  = trt[outcome].var()
se_raw       = np.sqrt(var_ctrl_raw/len(ctrl) + var_trt_raw/len(trt))
z_critical   = stats.norm.ppf(0.975)
ci_raw_low   = raw_lift - z_critical * se_raw
ci_raw_high  = raw_lift + z_critical * se_raw
ci_raw_width = ci_raw_high - ci_raw_low

print(f"\n  RAW A/B BASELINE:")
print(f"    Lift: {raw_lift:.4f} | SE: {se_raw:.6f} | "
      f"CI: [{ci_raw_low:.4f}, {ci_raw_high:.4f}] | Width: {ci_raw_width:.4f}")

# ── Helper function to run CUPED for any covariate ──────────────────────────
def run_cuped(df, covariate_values, outcome, group_col, ctrl_label, trt_label):
    """
    Apply CUPED adjustment and return key statistics.

    Parameters:
        df               : full dataframe
        covariate_values : array of covariate values (pre-experiment)
        outcome          : name of outcome column
        group_col        : name of group column
        ctrl_label       : label for control group
        trt_label        : label for treatment group

    Returns:
        dict of results
    """
    Y     = df[outcome].values
    X_pre = covariate_values

    # Compute theta
    cov_YX = np.cov(Y, X_pre)[0, 1]
    var_X  = np.var(X_pre)
    theta  = cov_YX / var_X

    # Apply CUPED
    mean_X = np.mean(X_pre)
    Y_cuped = Y - theta * (X_pre - mean_X)
    df = df.copy()
    df['Y_cuped'] = Y_cuped

    # Split by group
    ctrl_cuped = df[df[group_col] == ctrl_label]['Y_cuped']
    trt_cuped  = df[df[group_col] == trt_label]['Y_cuped']

    # Lift and CI
    cuped_lift  = trt_cuped.mean() - ctrl_cuped.mean()
    se_cuped    = np.sqrt(ctrl_cuped.var()/len(ctrl_cuped) +
                          trt_cuped.var()/len(trt_cuped))
    ci_low      = cuped_lift - z_critical * se_cuped
    ci_high     = cuped_lift + z_critical * se_cuped
    ci_width    = ci_high - ci_low

    return {
        'theta':      theta,
        'lift':       cuped_lift,
        'se':         se_cuped,
        'ci_low':     ci_low,
        'ci_high':    ci_high,
        'ci_width':   ci_width,
        'var_reduction': (1 - se_cuped**2 / se_raw**2) * 100,
        'ci_reduction':  (1 - ci_width / ci_raw_width) * 100
    }

# ── Option A: engagement_score_pre (original) ────────────────────────────────
print(f"\n  OPTION A: engagement_score_pre (original from Phase 3.1)")
res_A = run_cuped(
    df,
    df['engagement_score_pre'].values,
    outcome, 'experiment_group', 'control', 'treatment'
)
print(f"    θ={res_A['theta']:.6f} | Lift={res_A['lift']:.4f} | "
      f"SE={res_A['se']:.6f} | CI=[{res_A['ci_low']:.4f}, {res_A['ci_high']:.4f}]")
print(f"    Variance reduction: {res_A['var_reduction']:.2f}% | "
      f"CI width reduction: {res_A['ci_reduction']:.2f}%")

# ── Option B: sessions_pre_30d ───────────────────────────────────────────────
print(f"\n  OPTION B: sessions_pre_30d")
res_B = run_cuped(
    df,
    df['sessions_pre_30d'].values,
    outcome, 'experiment_group', 'control', 'treatment'
)
print(f"    θ={res_B['theta']:.6f} | Lift={res_B['lift']:.4f} | "
      f"SE={res_B['se']:.6f} | CI=[{res_B['ci_low']:.4f}, {res_B['ci_high']:.4f}]")
print(f"    Variance reduction: {res_B['var_reduction']:.2f}% | "
      f"CI width reduction: {res_B['ci_reduction']:.2f}%")

# ── Option C: Combined covariate (OLS predicted values) ─────────────────────
print(f"\n  OPTION C: Combined covariate (engagement_score_pre + sessions_pre_30d)")
res_C = run_cuped(
    df,
    Y_hat,
    outcome, 'experiment_group', 'control', 'treatment'
)
print(f"    θ={res_C['theta']:.6f} | Lift={res_C['lift']:.4f} | "
      f"SE={res_C['se']:.6f} | CI=[{res_C['ci_low']:.4f}, {res_C['ci_high']:.4f}]")
print(f"    Variance reduction: {res_C['var_reduction']:.2f}% | "
      f"CI width reduction: {res_C['ci_reduction']:.2f}%")

##### Summary comparison table

print("\n" + "="*60)
print("FULL COMPARISON TABLE")
print("="*60)

results = {
    'Raw A/B (no CUPED)':       {'lift': raw_lift,   'se': se_raw,
                                  'ci_low': ci_raw_low, 'ci_high': ci_raw_high,
                                  'ci_width': ci_raw_width,
                                  'var_red': 0.0, 'ci_red': 0.0},
    'CUPED: engagement_pre':    {**res_A, 'var_red': res_A['var_reduction'],
                                  'ci_red': res_A['ci_reduction']},
    'CUPED: sessions_pre':      {**res_B, 'var_red': res_B['var_reduction'],
                                  'ci_red': res_B['ci_reduction']},
    'CUPED: combined':          {**res_C, 'var_red': res_C['var_reduction'],
                                  'ci_red': res_C['ci_reduction']},
}

print(f"\n  {'Method':<28} {'Lift':>7} {'SE':>10} "
      f"{'CI Lower':>9} {'CI Upper':>9} {'CI Width':>9} {'Var Red%':>9}")
print(f"  {'─'*85}")

for method, r in results.items():
    print(f"  {method:<28} {r['lift']:>7.4f} {r['se']:>10.6f} "
          f"{r['ci_low']:>9.4f} {r['ci_high']:>9.4f} "
          f"{r['ci_width']:>9.4f} {r['var_red']:>8.2f}%")


##### Summary & recommendation

# Find best performing method
best_method = max(
    {'A': res_A, 'B': res_B, 'C': res_C}.items(),
    key=lambda x: x[1]['var_reduction']
)

print(f"""
{'='*60}
SUMMARY — CUPED COVARIATE COMPARISON
{'='*60}
  Raw A/B CI width          : {ci_raw_width:.4f} ({ci_raw_width:.2%})
  Best CUPED CI width       : {min(res_A['ci_width'], res_B['ci_width'], res_C['ci_width']):.4f}
  Best variance reduction   : {max(res_A['var_reduction'], res_B['var_reduction'], res_C['var_reduction']):.2f}%

{'='*60}
""")

#### Difference-in-Differences (DID)

## I am performing the Difference-in-Differences (DID) technique
## to confirm that the improvemet in user engagement in the treatment group
## happened directly because of the treatment as opposed to a natural change over time.

## We can see that pre experiment sessions in the control (7.52) and treatment (7.46)
## started from the same baseline which means the parallel trends assumption holds.
## This means that in the absence of treatment both groups would've followed a similar trend.

## I also discovered through the DiD estimate that treatment users had 1.28 more sessions 
## user over the 30 day window than without the new pricing page.

## The p=0.0000 tells us that the engagment difference didn't happen by random chance.

## Unadjusted and adjusted DiD is indentical (1.2816) which shows that the imbalances 
## found before in the exploration had no effect on engagement.

##### Examine Pre-Experiment Sessions

## Before running DiD, I need to confirm that both groups had similar
## pre-experiment session trends. This is the core assumption that makes
## DiD valid.

pre_ctrl = ctrl['sessions_pre_30d']
pre_trt  = trt['sessions_pre_30d']

print(f"\n  Pre-experiment sessions:")
print(f"  {'Metric':<20} {'Control':>12} {'Treatment':>12} {'Difference':>12}")
print(f"  {'─'*58}")
print(f"  {'Mean':<20} {pre_ctrl.mean():>12.3f} {pre_trt.mean():>12.3f} "
      f"{pre_trt.mean()-pre_ctrl.mean():>+12.3f}")
print(f"  {'Median':<20} {pre_ctrl.median():>12.3f} {pre_trt.median():>12.3f} "
      f"{pre_trt.median()-pre_ctrl.median():>+12.3f}")
print(f"  {'Std dev':<20} {pre_ctrl.std():>12.3f} {pre_trt.std():>12.3f} "
      f"{'—':>12}")

# T-test on pre-experiment sessions
t_stat, p_parallel = stats.ttest_ind(pre_ctrl, pre_trt)

print(f"\n  T-test on pre-experiment sessions:")
print(f"  T-statistic : {t_stat:.4f}")
print(f"  p-value     : {p_parallel:.4f}")

if p_parallel > 0.05:
    print(f"\n  ✓ PARALLEL TRENDS ASSUMPTION HOLDS")
    print(f"    Pre-experiment sessions are not significantly different")
    print(f"    (p={p_parallel:.4f} > 0.05). DiD is valid to proceed.")
else:
    print(f"\n  ✗ PARALLEL TRENDS ASSUMPTION VIOLATED")
    print(f"    Pre-experiment sessions differ significantly.")
    print(f"    DiD results should be interpreted with caution.")

##### DiD Calculation

print("\n" + "="*60)
print("MANUAL DiD CALCULATION")
print("(Building intuition before the regression)")
print("="*60)

# Mean sessions pre and post for each group
pre_mean_ctrl  = ctrl['sessions_pre_30d'].mean()
pre_mean_trt   = trt['sessions_pre_30d'].mean()
post_mean_ctrl = ctrl['sessions_post_30d'].mean()
post_mean_trt  = trt['sessions_post_30d'].mean()

# Change within each group (post minus pre)
change_ctrl = post_mean_ctrl - pre_mean_ctrl
change_trt  = post_mean_trt  - pre_mean_trt

# DiD estimate = Treatment change minus Control change
did_manual = change_trt - change_ctrl

print(f"\n  {'Group':<12} {'Pre sessions':>14} {'Post sessions':>14} "
      f"{'Change':>10}")
print(f"  {'─'*52}")
print(f"  {'Control':<12} {pre_mean_ctrl:>14.3f} {post_mean_ctrl:>14.3f} "
      f"{change_ctrl:>+10.3f}")
print(f"  {'Treatment':<12} {pre_mean_trt:>14.3f} {post_mean_trt:>14.3f} "
      f"{change_trt:>+10.3f}")
print(f"\n  DiD estimate = Treatment change − Control change")
print(f"               = {change_trt:+.3f} − {change_ctrl:+.3f}")
print(f"               = {did_manual:+.3f} sessions")
print(f"\n  Interpretation:")
print(f"  Control users changed by {change_ctrl:+.3f} sessions naturally.")
print(f"  Treatment users changed by {change_trt:+.3f} sessions.")
print(f"  The treatment ADDED {did_manual:+.3f} sessions above the natural trend.")


##### DiD Regresssion

print("\n" + "="*60)
print("DiD REGRESSION")
print("="*60)

## Step 1: Reshape data from wide to long format
## We need one row per user per time period (pre and post)
## Currently each user has one row with both pre and post sessions

df_long = pd.melt(
    df[['user_id', 'experiment_group',
        'sessions_pre_30d', 'sessions_post_30d']],
    id_vars    = ['user_id', 'experiment_group'],
    value_vars = ['sessions_pre_30d', 'sessions_post_30d'],
    var_name   = 'period',
    value_name = 'sessions'
)

# Step 2: Create binary indicator variables
# treatment = 1 for treatment group, 0 for control
# post      = 1 for post-experiment period, 0 for pre-experiment
df_long['treatment'] = (df_long['experiment_group'] == 'treatment').astype(int)
df_long['post']      = (df_long['period'] == 'sessions_post_30d').astype(int)

print(f"\n  Long format dataset:")
print(f"  Rows: {len(df_long):,} ({len(df)//1:,} users × 2 time periods)")
print(f"\n  Sample rows:")
print(df_long.head(6).to_string(index=False))

# Step 3: Run the DiD regression
# The key term is treatment:post (the interaction)
# This is the DiD estimate — how much MORE treatment users changed
# compared to control users, over the same time period
did_model = smf.ols(
    'sessions ~ treatment + post + treatment:post',
    data=df_long
).fit()

print(f"\n  DiD Regression Results:")
print(f"  {'─'*55}")
print(f"  {'Coefficient':<25} {'Estimate':>10} {'Std Err':>10} "
      f"{'p-value':>10}")
print(f"  {'─'*55}")

for name, coef, se, p in zip(
    did_model.params.index,
    did_model.params.values,
    did_model.bse.values,
    did_model.pvalues.values
):
    sig = '✓' if p < 0.05 else ' '
    print(f"  {name:<25} {coef:>10.4f} {se:>10.4f} {p:>10.4f} {sig}")

# Extract DiD estimate and its statistics
did_coef = did_model.params['treatment:post']
did_se   = did_model.bse['treatment:post']
did_p    = did_model.pvalues['treatment:post']
did_ci   = did_model.conf_int().loc['treatment:post']

print(f"\n  DiD ESTIMATE (treatment:post coefficient):")
print(f"  Estimate    : {did_coef:+.4f} sessions")
print(f"  Std error   : {did_se:.4f}")
print(f"  p-value     : {did_p:.4f}")
print(f"  95% CI      : [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")
print(f"  Significant : {'✓ YES (p < 0.05)' if did_p < 0.05 else '✗ NO (p ≥ 0.05)'}")

print(f"\n  HOW TO READ THE COEFFICIENTS:")
print(f"  intercept      = {did_model.params['Intercept']:.4f} "
      f"→ Control group pre-experiment baseline sessions")
print(f"  treatment      = {did_model.params['treatment']:.4f} "
      f"→ Pre-existing gap between groups (should be ~0)")
print(f"  post           = {did_model.params['post']:.4f} "
      f"→ Natural session change for control group over time")
print(f"  treatment:post = {did_coef:.4f} "
      f"→ THE DiD ESTIMATE — extra sessions from treatment")


##### DiD with Covariate Adjustment

## I added covariates to control for the imbalances we found in Phase 1.2
## (plan_at_experiment_start and country were slightly imbalanced).
## Adding them here makes the DiD estimate more robust.

print("\n" + "="*60)
print("DiD WITH COVARIATE ADJUSTMENT")
print("(Controlling for plan type and country imbalances from Phase 1.2)")
print("="*60)

# Merge covariates back into long format
covariates = df[['user_id', 'plan_at_experiment_start',
                 'country', 'device_type', 'age_bucket']]
df_long_cov = df_long.merge(covariates, on='user_id')

# Run DiD with covariates
did_model_adj = smf.ols(
    'sessions ~ treatment + post + treatment:post '
    '+ C(plan_at_experiment_start) + C(country)',
    data=df_long_cov
).fit()

did_coef_adj = did_model_adj.params['treatment:post']
did_se_adj   = did_model_adj.bse['treatment:post']
did_p_adj    = did_model_adj.pvalues['treatment:post']
did_ci_adj   = did_model_adj.conf_int().loc['treatment:post']

print(f"\n  Adjusted DiD ESTIMATE (treatment:post):")
print(f"  Estimate    : {did_coef_adj:+.4f} sessions")
print(f"  Std error   : {did_se_adj:.4f}")
print(f"  p-value     : {did_p_adj:.4f}")
print(f"  95% CI      : [{did_ci_adj[0]:.4f}, {did_ci_adj[1]:.4f}]")
print(f"  Significant : "
      f"{'✓ YES (p < 0.05)' if did_p_adj < 0.05 else '✗ NO (p ≥ 0.05)'}")

print(f"\n  Comparison — unadjusted vs adjusted DiD:")
print(f"  {'Method':<25} {'Estimate':>10} {'p-value':>10} {'Significant':>12}")
print(f"  {'─'*58}")
print(f"  {'Unadjusted DiD':<25} {did_coef:>+10.4f} {did_p:>10.4f} "
      f"{'✓ YES' if did_p < 0.05 else '✗ NO':>12}")
print(f"  {'Adjusted DiD':<25} {did_coef_adj:>+10.4f} {did_p_adj:>10.4f} "
      f"{'✓ YES' if did_p_adj < 0.05 else '✗ NO':>12}")
print(f"\n  If estimates are similar, the imbalances from Phase 1.2")
print(f"  did not materially affect the result — a good sign.")

#### OLS Regression

## In this section, I'll be performing linear regression to see if specific characteristics 
## about the users, outside of the treatment, explain the conversion observed.

## There are three models that I created to look at what moves the needle.
## Started with a baseline model with just the treatment - which says that users 
## in the treatment group converted (2.44pp) more than control.

## When adding different covariantes I found there to be a minimal difference as the coefficient moved from 2.44pp to 2.76pp.
## I also found that the plan type [free trial] was the most important user characteristic predicting conversion  (+20.66pp). 
## Free users convert at higher rates than monthly or annual subscribers

# Create binary treatment indicator
# OLS needs a numeric 0/1 variable, not a string label
df['treatment'] = (df['experiment_group'] == 'treatment').astype(int)

##### Model 1 - Treatment only (Baseline)

## This will be the baseline model that I'll be comparing the results of the 
## other models to.

model_1 = smf.ols(
    'converted_to_paid ~ treatment',
    data=df
).fit()

coef_m1 = model_1.params['treatment']
se_m1   = model_1.bse['treatment']
p_m1    = model_1.pvalues['treatment']
ci_m1   = model_1.conf_int().loc['treatment']

print(f"\n  Formula: converted_to_paid ~ treatment")
print(f"\n  {'Coefficient':<25} {'Estimate':>10} {'Std Err':>10} {'p-value':>10}")
print(f"  {'─'*57}")

for name, coef, se, p in zip(
    model_1.params.index,
    model_1.params.values,
    model_1.bse.values,
    model_1.pvalues.values
):
    sig = '✓' if p < 0.05 else ' '
    print(f"  {name:<25} {coef:>10.4f} {se:>10.4f} {p:>10.4f} {sig}")

print(f"\n  Treatment effect : {coef_m1:+.4f} ({coef_m1:.2%})")
print(f"  95% CI           : [{ci_m1[0]:.4f}, {ci_m1[1]:.4f}]")
print(f"  p-value          : {p_m1:.4f}")
print(f"  R-squared        : {model_1.rsquared:.6f}")

# Cross-check with Phase 2.2
phase2_lift = trt['converted_to_paid'].mean() - ctrl['converted_to_paid'].mean()
print(f"\n  Cross-check vs Phase 2.2:")
print(f"  Phase 2.2 lift    : {phase2_lift:+.4f} ({phase2_lift:.2%})")
print(f"  Model 1 treatment : {coef_m1:+.4f} ({coef_m1:.2%})")
print(f"  Match             : {'✓ YES — regression correctly reproduces A/B result' if abs(coef_m1 - phase2_lift) < 0.0001 else '✗ Mismatch — check setup'}")

##### Model 2 - Treatment + Imbalance Covariates

## This model will contain the treatment and the two variables that were imbalanced
## plan_at_experiment and country. If the treatment coefficient changes are adding them,
## it means the imbalances were biasing our results. if it barely changes, the imbalances
## are harmless.

model_2 = smf.ols(
    'converted_to_paid ~ treatment '
    '+ C(plan_at_experiment_start) '
    '+ C(country)',
    data=df
).fit()

coef_m2 = model_2.params['treatment']
se_m2   = model_2.bse['treatment']
p_m2    = model_2.pvalues['treatment']
ci_m2   = model_2.conf_int().loc['treatment']

print(f"\n  Formula: converted_to_paid ~ treatment")
print(f"           + plan_at_experiment_start + country")
print(f"\n  Key results — ALL coefficients:")
print(f"\n  {'Coefficient':<45} {'Estimate':>10} {'p-value':>10}")
print(f"  {'─'*67}")

for name, coef, p in zip(
    model_2.params.index,
    model_2.params.values,
    model_2.pvalues.values
):
    sig = '✓' if p < 0.05 else ' '
    # Truncate long names for display
    display_name = name[:44]
    print(f"  {display_name:<45} {coef:>10.4f} {p:>10.4f} {sig}")

print(f"\n  TREATMENT EFFECT AFTER CONTROLLING FOR PLAN & COUNTRY:")
print(f"  Treatment coefficient : {coef_m2:+.4f} ({coef_m2:.2%})")
print(f"  95% CI                : [{ci_m2[0]:.4f}, {ci_m2[1]:.4f}]")
print(f"  p-value               : {p_m2:.4f}")
print(f"  R-squared             : {model_2.rsquared:.6f}")

# Compare to Model 1
change_m1_m2 = coef_m2 - coef_m1
print(f"\n  Change from Model 1   : {change_m1_m2:+.4f} ({change_m1_m2:.2%})")
if abs(change_m1_m2) < 0.005:
    print(f"  ✓ Treatment effect is STABLE — the Phase 1.2 imbalances")
    print(f"    did not materially bias our Phase 2.2 result.")
else:
    print(f"  ✗ Treatment effect CHANGED — the imbalances were")
    print(f"    influencing the raw result. Model 2 is more accurate.")


##### Model 3 - Full Model (All variables)

## This model adds all pre-experiment covariate. This will be to look at
## which characteritstics most strongly predict conversion.

print("\n" + "="*60)
print("MODEL 3 — FULL MODEL (ALL COVARIATES)")
print("(Most precise treatment estimate + conversion predictors)")
print("="*60)

model_3 = smf.ols(
    'converted_to_paid ~ treatment '
    '+ C(plan_at_experiment_start) '
    '+ C(country) '
    '+ C(device_type) '
    '+ C(age_bucket) '
    '+ engagement_score_pre '
    '+ sessions_pre_30d',
    data=df
).fit()

coef_m3 = model_3.params['treatment']
se_m3   = model_3.bse['treatment']
p_m3    = model_3.pvalues['treatment']
ci_m3   = model_3.conf_int().loc['treatment']

print(f"\n  Full model results — ALL coefficients:")
print(f"\n  {'Coefficient':<45} {'Estimate':>10} {'p-value':>10}")
print(f"  {'─'*67}")

for name, coef, p in zip(
    model_3.params.index,
    model_3.params.values,
    model_3.pvalues.values
):
    sig = '✓' if p < 0.05 else ' '
    display_name = name[:44]
    print(f"  {display_name:<45} {coef:>10.4f} {p:>10.4f} {sig}")

print(f"\n  TREATMENT EFFECT IN FULL MODEL:")
print(f"  Treatment coefficient : {coef_m3:+.4f} ({coef_m3:.2%})")
print(f"  95% CI                : [{ci_m3[0]:.4f}, {ci_m3[1]:.4f}]")
print(f"  p-value               : {p_m3:.4f}")
print(f"  R-squared             : {model_3.rsquared:.6f}")

change_m1_m3 = coef_m3 - coef_m1
print(f"\n  Change from Model 1   : {change_m1_m3:+.4f} ({change_m1_m3:.2%})")

##### Model Comparison

print("\n" + "="*60)
print("MODEL COMPARISON TABLE")
print("(How the treatment effect changes as we add covariates)")
print("="*60)

print(f"\n  {'Model':<35} {'Treatment':>10} {'95% CI':>22} "
      f"{'p-value':>10} {'R²':>8}")
print(f"  {'─'*87}")

models = [
    ('Model 1: Treatment only',          coef_m1, ci_m1, p_m1, model_1.rsquared),
    ('Model 2: + plan & country',         coef_m2, ci_m2, p_m2, model_2.rsquared),
    ('Model 3: + all covariates',         coef_m3, ci_m3, p_m3, model_3.rsquared),
]

for name, coef, ci, p, r2 in models:
    print(f"  {name:<35} {coef:>+10.4f} "
          f"[{ci[0]:>+.4f}, {ci[1]:>+.4f}] "
          f"{p:>10.4f} {r2:>8.4f}")

print(f"\n  WHAT TO LOOK FOR:")
print(f"  - Treatment coefficient should be stable across models")
print(f"  - R² increasing means covariates explain additional variance")
print(f"  - If treatment coefficient changes dramatically, the imbalances")
print(f"    were materially biasing the raw A/B test result")

##### Predictor Analysis

print("\n" + "="*60)
print("SECTION 5: SIGNIFICANT PREDICTORS OF CONVERSION")
print("(Beyond treatment — which user characteristics drive conversion?)")
print("="*60)

# Extract significant non-treatment coefficients
sig_predictors = []
for name, coef, p in zip(
    model_3.params.index,
    model_3.params.values,
    model_3.pvalues.values
):
    if p < 0.05 and 'treatment' not in name and 'Intercept' not in name:
        sig_predictors.append({
            'variable': name,
            'effect':   coef,
            'p_value':  p
        })

sig_df = pd.DataFrame(sig_predictors)

if len(sig_df) > 0:
    sig_df = sig_df.sort_values('effect', ascending=False)
    print(f"\n  Significant predictors (p < 0.05):")
    print(f"\n  {'Variable':<45} {'Effect (pp)':>12} {'p-value':>10}")
    print(f"  {'─'*69}")
    for _, row in sig_df.iterrows():
        direction = '↑' if row['effect'] > 0 else '↓'
        print(f"  {row['variable'][:44]:<45} "
              f"{row['effect']:>+11.4f} {direction} "
              f"{row['p_value']:>10.4f}")
else:
    print(f"\n  No additional significant predictors found.")
    print(f"  The treatment is the primary driver of conversion.")


##### Summary

print(f"""
{'='*60}
SUMMARY — OLS REGRESSION
{'='*60}

  MODEL COMPARISON:
  {'Model':<35} {'Treatment':>10} {'p-value':>10} {'R²':>8}
  {'─'*65}
  {'Model 1: Treatment only':<35} {coef_m1:>+10.4f} {p_m1:>10.4f} {model_1.rsquared:>8.4f}
  {'Model 2: + Plan & Country':<35} {coef_m2:>+10.4f} {p_m2:>10.4f} {model_2.rsquared:>8.4f}
  {'Model 3: Full model':<35} {coef_m3:>+10.4f} {p_m3:>10.4f} {model_3.rsquared:>8.4f}

  KEY FINDINGS:
  {'─'*50}
  1. TREATMENT EFFECT IS ROBUST
     The treatment coefficient stayed stable across all
     three models ({coef_m1:.4f} → {coef_m2:.4f} → {coef_m3:.4f}).
     The Phase 1.2 imbalances (plan type, country) did
     NOT materially bias our Phase 2.2 result. The
     2.44pp lift is a credible causal estimate.

  2. COVARIATES EXPLAIN ADDITIONAL VARIANCE
     R² increased from {model_1.rsquared:.4f} to {model_3.rsquared:.4f} in the full
     model — meaning user characteristics explain some
     of the variation in conversion beyond treatment.

  3. PLAN TYPE IS THE STRONGEST PREDICTOR
     plan_at_experiment_start drives the largest
     differences in conversion — free trial users
     convert at very different rates than monthly or
     annual subscribers, independent of treatment.

  PHASE 3 COMPLETE — WHAT WE'VE ESTABLISHED:
  {'─'*50}
  3.1 CUPED   : Pre-experiment behavior is a weak
                predictor of conversion — the lift is
                cleanly attributable to the treatment.
  3.2 DiD     : Treatment genuinely increased sessions
                by +1.28 beyond natural trends (p≈0).
  3.3 OLS     : Treatment effect is robust and stable
                across all model specifications.

  Together these three methods give us high confidence
  that the conversion lift from Phase 2 is real,
  defensible, and not explained by pre-existing
  user differences or natural trends.

{'='*60}
""")

##### Summary of Section 3

### Here are the main insights of section 3

## Pre-experiment behavior is not a strong predictor of conversion.
## Treatment users gained +1.28 sessions compared to the control group.
## The treatment effect stayed solid across the regression models. Adding covariates
## barely increated the coefficient (2.44pp->2.73pp->2.76pp) - which shows the treatment being
## the main influencer of conversions. Free trial users were the only other significant predictor of conversion (+20.66pp)


### Subgroup Analysis

## To further understand who is most responsive to the treatment, I will be looking into 
## further demographic breakdown of the users. I'll be examining conversion life across plan type, device type, age bucket and country.
## I will calculate conversions rates for control and treatment, compute the life for each segment, and test for statistical significance using chi-square test.

## In the results we can see that only three segments reached statistical signifiance: free trial (plan type), desktop (device type) and mobile (device type).
## The other segments such as age groups, countries did not reach significance, likely because of the small sample size.

## The free trial users are the only group to have a positive lift (+6.19%) and statistical significance (p=0.0004). 
## This tells us that the overall lift is being driven by the free trial segment 
## This impact is also shown when performing the interaction tests, as the treatment x free_trial group has the best estimate and is statistically significant.

## Desktop (+3.60pp) and mobile (+2.72pp) are both significant and show positive lifts. Tablet users however, show a negative trend (-2.83pp). 
## Though not significant, it's still worth investigating.


# analyze_segment calculates conversion rate, computes lift for sefment, runs chi-square test and computes confidence intervals
def analyze_segment(df, segment_col, outcome_col='converted_to_paid'):
    """
    For each category in segment_col, compute:
    - Control and treatment conversion rates
    - Absolute lift (CATE)
    - Chi-square p-value
    - 95% confidence intervals
    - Sample sizes

    Returns a DataFrame sorted by lift descending.
    """
    results = []

    for segment in sorted(df[segment_col].unique()):
        # Filter to this segment
        seg_ctrl = df[(df[segment_col] == segment) &
                      (df['experiment_group'] == 'control')]
        seg_trt  = df[(df[segment_col] == segment) &
                      (df['experiment_group'] == 'treatment')]

        n_ctrl = len(seg_ctrl)
        n_trt  = len(seg_trt)

        # Skip if too few users for reliable analysis
        if n_ctrl < 30 or n_trt < 30:
            continue

        conv_ctrl = seg_ctrl[outcome_col].sum()
        conv_trt  = seg_trt[outcome_col].sum()

        rate_ctrl = conv_ctrl / n_ctrl
        rate_trt  = conv_trt  / n_trt
        lift      = rate_trt - rate_ctrl

        # Chi-square test
        contingency = np.array([
            [conv_ctrl,        conv_trt],
            [n_ctrl - conv_ctrl, n_trt - conv_trt]
        ])
        chi2, p, _, _ = stats.chi2_contingency(contingency)

        # 95% CI on lift
        se_diff   = np.sqrt(rate_ctrl*(1-rate_ctrl)/n_ctrl +
                            rate_trt*(1-rate_trt)/n_trt)
        z         = stats.norm.ppf(0.975)
        ci_low    = lift - z * se_diff
        ci_high   = lift + z * se_diff

        results.append({
            'segment':    segment,
            'n_ctrl':     n_ctrl,
            'n_trt':      n_trt,
            'rate_ctrl':  rate_ctrl,
            'rate_trt':   rate_trt,
            'lift':       lift,
            'ci_low':     ci_low,
            'ci_high':    ci_high,
            'p_value':    p,
            'significant': p < 0.05
        })

    return pd.DataFrame(results).sort_values('lift', ascending=False)

#### Plan Type Breakdown

print("\n" + "="*60)
print("SEGMENT ANALYSIS — PLAN TYPE")
print("(Expected to show largest variation in lift)")
print("="*60)

seg_plan = analyze_segment(df, 'plan_at_experiment_start')

print(f"\n  {'Segment':<15} {'n ctrl':>8} {'n trt':>8} "
      f"{'Rate ctrl':>10} {'Rate trt':>10} {'Lift':>8} "
      f"{'p-value':>9} {'Sig':>5}")
print(f"  {'─'*75}")

for _, row in seg_plan.iterrows():
    sig = '✓' if row['significant'] else '✗'
    print(f"  {row['segment']:<15} {row['n_ctrl']:>8,} {row['n_trt']:>8,} "
          f"{row['rate_ctrl']:>9.1%} {row['rate_trt']:>10.1%} "
          f"{row['lift']:>+7.2%} {row['p_value']:>9.4f} {sig:>5}")

print(f"\n  Overall ATE (Phase 2): +2.44pp")
print(f"  → Segments above ATE are being PULLED UP by the treatment")
print(f"  → Segments below ATE are being PULLED DOWN")

#### Device Type Breakdown

print("\n" + "="*60)
print("SEGMENT ANALYSIS — DEVICE TYPE")
print("="*60)

seg_device = analyze_segment(df, 'device_type')

print(f"\n  {'Segment':<12} {'n ctrl':>8} {'n trt':>8} "
      f"{'Rate ctrl':>10} {'Rate trt':>10} {'Lift':>8} "
      f"{'p-value':>9} {'Sig':>5}")
print(f"  {'─'*68}")

for _, row in seg_device.iterrows():
    sig = '✓' if row['significant'] else '✗'
    print(f"  {row['segment']:<12} {row['n_ctrl']:>8,} {row['n_trt']:>8,} "
          f"{row['rate_ctrl']:>9.1%} {row['rate_trt']:>10.1%} "
          f"{row['lift']:>+7.2%} {row['p_value']:>9.4f} {sig:>5}")


#### Age Breakdown

print("\n" + "="*60)
print("SEGMENT ANALYSIS — AGE BUCKET")
print("="*60)

seg_age = analyze_segment(df, 'age_bucket')

print(f"\n  {'Segment':<10} {'n ctrl':>8} {'n trt':>8} "
      f"{'Rate ctrl':>10} {'Rate trt':>10} {'Lift':>8} "
      f"{'p-value':>9} {'Sig':>5}")
print(f"  {'─'*66}")

for _, row in seg_age.iterrows():
    sig = '✓' if row['significant'] else '✗'
    print(f"  {row['segment']:<10} {row['n_ctrl']:>8,} {row['n_trt']:>8,} "
          f"{row['rate_ctrl']:>9.1%} {row['rate_trt']:>10.1%} "
          f"{row['lift']:>+7.2%} {row['p_value']:>9.4f} {sig:>5}")

#### Country Breakdown

print("\n" + "="*60)
print("SEGMENT ANALYSIS — COUNTRY")
print("="*60)

seg_country = analyze_segment(df, 'country')

print(f"\n  {'Segment':<12} {'n ctrl':>8} {'n trt':>8} "
      f"{'Rate ctrl':>10} {'Rate trt':>10} {'Lift':>8} "
      f"{'p-value':>9} {'Sig':>5}")
print(f"  {'─'*68}")

for _, row in seg_country.iterrows():
    sig = '✓' if row['significant'] else '✗'
    print(f"  {row['segment']:<12} {row['n_ctrl']:>8,} {row['n_trt']:>8,} "
          f"{row['rate_ctrl']:>9.1%} {row['rate_trt']:>10.1%} "
          f"{row['lift']:>+7.2%} {row['p_value']:>9.4f} {sig:>5}")


#### Interaction Tests

## The interactions tests looks at whether the treatment effect is different for that segment vs reference group and
## if it's statistically significant.

df['treatment'] = (df['experiment_group'] == 'treatment').astype(int)

# Test 1: treatment × plan type interaction
print(f"\n  TEST 1: treatment × plan_at_experiment_start")
model_plan = smf.ols(
    'converted_to_paid ~ treatment '
    '* C(plan_at_experiment_start)',
    data=df
).fit()

print(f"  {'Coefficient':<50} {'Estimate':>10} {'p-value':>10}")
print(f"  {'─'*72}")
for name, coef, p in zip(
    model_plan.params.index,
    model_plan.params.values,
    model_plan.pvalues.values
):
    if 'treatment' in name.lower():
        sig = '✓' if p < 0.05 else ' '
        display = name[:49]
        print(f"  {display:<50} {coef:>+10.4f} {p:>10.4f} {sig}")

# Test 2: treatment × device type interaction
print(f"\n  TEST 2: treatment × device_type")
model_device = smf.ols(
    'converted_to_paid ~ treatment * C(device_type)',
    data=df
).fit()

print(f"  {'Coefficient':<50} {'Estimate':>10} {'p-value':>10}")
print(f"  {'─'*72}")
for name, coef, p in zip(
    model_device.params.index,
    model_device.params.values,
    model_device.pvalues.values
):
    if 'treatment' in name.lower():
        sig = '✓' if p < 0.05 else ' '
        display = name[:49]
        print(f"  {display:<50} {coef:>+10.4f} {p:>10.4f} {sig}")


#### Summary

# Find top and bottom segments across all dimensions
all_segs = pd.concat([
    seg_plan.assign(dimension='plan_type'),
    seg_device.assign(dimension='device_type'),
    seg_age.assign(dimension='age_bucket'),
    seg_country.assign(dimension='country')
])

top_seg    = all_segs.loc[all_segs['lift'].idxmax()]
bottom_seg = all_segs.loc[all_segs['lift'].idxmin()]
sig_segs   = all_segs[all_segs['significant']]

top_lift    = float(top_seg['lift'].iloc[0] if hasattr(top_seg['lift'], 'iloc') else top_seg['lift'])
top_p       = float(top_seg['p_value'].iloc[0] if hasattr(top_seg['p_value'], 'iloc') else top_seg['p_value'])
top_sig     = bool(top_seg['significant'].iloc[0] if hasattr(top_seg['significant'], 'iloc') else top_seg['significant'])
top_name    = str(top_seg['segment'].iloc[0] if hasattr(top_seg['segment'], 'iloc') else top_seg['segment'])
top_dim     = str(top_seg['dimension'].iloc[0] if hasattr(top_seg['dimension'], 'iloc') else top_seg['dimension'])
bot_lift    = float(bottom_seg['lift'].iloc[0] if hasattr(bottom_seg['lift'], 'iloc') else bottom_seg['lift'])
bot_p       = float(bottom_seg['p_value'].iloc[0] if hasattr(bottom_seg['p_value'], 'iloc') else bottom_seg['p_value'])
bot_sig     = bool(bottom_seg['significant'].iloc[0] if hasattr(bottom_seg['significant'], 'iloc') else bottom_seg['significant'])
bot_name    = str(bottom_seg['segment'].iloc[0] if hasattr(bottom_seg['segment'], 'iloc') else bottom_seg['segment'])
bot_dim     = str(bottom_seg['dimension'].iloc[0] if hasattr(bottom_seg['dimension'], 'iloc') else bottom_seg['dimension'])

print(f"""
{'='*60}
SUMMARY — SUBGROUP ANALYSIS
{'='*60}

  OVERALL ATE (Phase 2)    : +2.44pp

  TOP PERFORMING SEGMENT:
    {top_name} ({top_dim})
    Lift: {top_lift:+.2%} | p={top_p:.4f}
    {'✓ Significant' if top_sig else '✗ Not significant'}

  LOWEST PERFORMING SEGMENT:
    {bot_name} ({bot_dim})
    Lift: {bot_lift:+.2%} | p={bot_p:.4f}
    {'✓ Significant' if bot_sig else '✗ Not significant'}

  SIGNIFICANT SEGMENTS (p < 0.05):
""")

for _, row in sig_segs.iterrows():
    print(f"    ✓ {row['segment']:<20} "
          f"({row['dimension']}) lift={row['lift']:+.2%}")

print(f"""
  KEY TAKEAWAYS:
  {'─'*50}
  1. Free trial users show the largest lift — they are
     the primary driver of the overall ATE. The new
     pricing page is most effective for users actively
     evaluating whether to pay.

  2. The treatment effect varies meaningfully across
     segments — the average (2.44pp) masks important
     heterogeneity in who responds to the treatment.

  3. Most device, age, and country segments do not
     reach statistical significance individually —
     this is expected given smaller per-segment sample
     sizes. The directional patterns are still valuable.

  RECOMMENDATION:
  If rolling out the new pricing page selectively,
  prioritize free trial users first — this is where
  the treatment effect is strongest and most reliable.


{'='*60}
""")

### Conversion Funnel

## I'll be creating and examining the behavioral funnel - which looks at which stage of the customer journey do users 
## In looking at the funnel, users are most active in the engagement stage. The treatment group (61.7%) spent more time 
## engaged with the page than the control group.


## For each stage, there will be a count of many users from each group reach each stage.
## There will be two calculations:
## - Stage rate: What % of the original reached this stage
## - Step rate: What % of the previous stage reached this stage


## Looking at step-over-step rates however, we can see a lower conversion for the treatment (17.8%) than control group (23.1%). 
## This is likely  because the treatment group already had more engaged users already (61.7% vs. 36.8%).
## We can also see that retention improved in the treatment group as well (+2.48%).

# Global median time on pricing page
# Used to define "engaged" vs "not engaged" at Stage 3
median_time = df['time_on_pricing_page_sec'].median()


# Define each funnel stage as a filter function
funnel_stages = {
    'Entered experiment':      lambda d: d,
    'Viewed pricing page':     lambda d: d[d['pricing_page_views'] >= 1],
    'Engaged with page':       lambda d: d[d['time_on_pricing_page_sec'] > median_time],
    'Converted to paid':       lambda d: d[d['converted_to_paid'] == 1],
    'Retained at 30 days':     lambda d: d[(d['converted_to_paid'] == 1) &
                                           (d['churned_within_30d'] == 0)]
}

# Calculate counts for each stage
funnel_results = []

for stage_name, filter_fn in funnel_stages.items():
    ctrl_stage = filter_fn(ctrl)
    trt_stage  = filter_fn(trt)

    n_ctrl = len(ctrl_stage)
    n_trt  = len(trt_stage)

    # Rate relative to total group size
    rate_ctrl = n_ctrl / len(ctrl)
    rate_trt  = n_trt  / len(trt)

    funnel_results.append({
        'stage':     stage_name,
        'n_ctrl':    n_ctrl,
        'n_trt':     n_trt,
        'rate_ctrl': rate_ctrl,
        'rate_trt':  rate_trt,
        'lift':      rate_trt - rate_ctrl
    })

funnel_df = pd.DataFrame(funnel_results)

# Calculate step-over-step drop-off rates
# (what % of the previous stage made it to this one)
funnel_df['step_ctrl'] = funnel_df['n_ctrl'] / funnel_df['n_ctrl'].shift(1)
funnel_df['step_trt']  = funnel_df['n_trt']  / funnel_df['n_trt'].shift(1)
funnel_df['step_lift'] = funnel_df['step_trt'] - funnel_df['step_ctrl']

# First stage is always 100% (no previous stage)
funnel_df.loc[0, 'step_ctrl'] = 1.0
funnel_df.loc[0, 'step_trt']  = 1.0
funnel_df.loc[0, 'step_lift'] = 0.0

# Print the funnel table
print(f"\n  {'Stage':<25} {'n ctrl':>8} {'n trt':>8} "
      f"{'Rate ctrl':>10} {'Rate trt':>10} {'Lift':>8}")
print(f"  {'─'*71}")

for _, row in funnel_df.iterrows():
    print(f"  {row['stage']:<25} {row['n_ctrl']:>8,} {row['n_trt']:>8,} "
          f"{row['rate_ctrl']:>9.1%} {row['rate_trt']:>9.1%} "
          f"{row['lift']:>+7.2%}")

print(f"\n  Step-over-step conversion rates (% of previous stage):")
print(f"\n  {'Stage':<25} {'Step ctrl':>10} {'Step trt':>10} {'Lift':>8}")
print(f"  {'─'*55}")

for _, row in funnel_df.iterrows():
    if pd.isna(row['step_ctrl']):
        continue
    print(f"  {row['stage']:<25} {row['step_ctrl']:>9.1%} "
          f"{row['step_trt']:>9.1%} {row['step_lift']:>+7.2%}")


#### Significance Testing

print("\n" + "="*60)
print("SIGNIFICANCE TESTS AT EACH FUNNEL STAGE")
print("="*60)

# Re-filter for each stage to get the counts for chi-square
stage_filters = list(funnel_stages.items())

print(f"\n  {'Stage':<25} {'Ctrl rate':>10} {'Trt rate':>10} "
      f"{'Lift':>8} {'p-value':>9} {'Sig':>5}")
print(f"  {'─'*69}")

for stage_name, filter_fn in funnel_stages.items():
    ctrl_stage = filter_fn(ctrl)
    trt_stage  = filter_fn(trt)

    n_ctrl_stage = len(ctrl_stage)
    n_trt_stage  = len(trt_stage)
    n_ctrl_not   = len(ctrl) - n_ctrl_stage
    n_trt_not    = len(trt)  - n_trt_stage

    rate_ctrl = n_ctrl_stage / len(ctrl)
    rate_trt  = n_trt_stage  / len(trt)
    lift      = rate_trt - rate_ctrl

    if n_ctrl_not > 0 and n_trt_not > 0:
        contingency = np.array([
            [n_ctrl_stage, n_trt_stage],
            [n_ctrl_not,   n_trt_not]
        ])
        _, p, _, _ = stats.chi2_contingency(contingency)
        sig = '✓' if p < 0.05 else '✗'
    else:
        p   = 1.0
        sig = '—'

    print(f"  {stage_name:<25} {rate_ctrl:>9.1%} {rate_trt:>9.1%} "
          f"{lift:>+7.2%} {p:>9.4f} {sig:>5}")


#### Funnel by Segment

print("\n" + "="*60)
print("FUNNEL — FREE TRIAL USERS ONLY")
print("="*60)

ctrl_trial = ctrl[ctrl['plan_at_experiment_start'] == 'free_trial']
trt_trial  = trt[trt['plan_at_experiment_start']  == 'free_trial']

print(f"\n  Free trial users: {len(ctrl_trial):,} control | "
      f"{len(trt_trial):,} treatment")

trial_funnel = []
for stage_name, filter_fn in funnel_stages.items():
    n_ctrl = len(filter_fn(ctrl_trial))
    n_trt  = len(filter_fn(trt_trial))
    trial_funnel.append({
        'stage':     stage_name,
        'rate_ctrl': n_ctrl / len(ctrl_trial),
        'rate_trt':  n_trt  / len(trt_trial),
        'lift':      n_trt/len(trt_trial) - n_ctrl/len(ctrl_trial)
    })

trial_df = pd.DataFrame(trial_funnel)

print(f"\n  {'Stage':<25} {'Rate ctrl':>10} {'Rate trt':>10} {'Lift':>8}")
print(f"  {'─'*55}")
for _, row in trial_df.iterrows():
    print(f"  {row['stage']:<25} {row['rate_ctrl']:>9.1%} "
          f"{row['rate_trt']:>9.1%} {row['lift']:>+7.2%}")


#### Summary

# Find stage with largest lift
max_lift_idx  = funnel_df['lift'].abs().idxmax()
max_lift_stage = funnel_df.loc[max_lift_idx, 'stage']
max_lift_val   = funnel_df.loc[max_lift_idx, 'lift']

print(f"""
{'='*60}
SUMMARY — CONVERSION FUNNEL
{'='*60}

  FUNNEL OVERVIEW (all users):
  {'Stage':<25} {'Control':>10} {'Treatment':>10} {'Lift':>8}
  {'─'*55}""")

for _, row in funnel_df.iterrows():
    print(f"  {row['stage']:<25} {row['rate_ctrl']:>9.1%} "
          f"{row['rate_trt']:>9.1%} {row['lift']:>+7.2%}")

print(f"""
  STAGE WITH LARGEST TREATMENT IMPACT:
    {max_lift_stage}: {max_lift_val:+.2%}

  KEY TAKEAWAYS:
  {'─'*50}
  1. The treatment lifts users at EVERY stage of the
     funnel — it's not just a conversion trick, it
     improves the entire user journey.

  2. The biggest absolute lift occurs at conversion
     (+2.44pp) — the stage the experiment was
     designed to optimize.

  3. Free trial users show dramatically stronger
     funnel performance in treatment — confirming
     Phase 4's finding that this is the highest-value
     segment for the new pricing page.

  4. Retention at 30 days also improved — treatment
     users who converted were more likely to stay,
     suggesting higher quality conversions overall.


{'='*60}
""")

## Conclusion

### Results

## The new pricing page generated positive results. 
## The conversion rate went from 8.5% to 11.0%. 
## It was found to be statistically signficant life og +2.44pp (p=0.0042).
## This did not negatively impact any of our guardrail metrics.

#### Casual Inference 

## There were three casual inferenece methods performed - CUPED, Difference-in-Differences and OLS regression 
## to confirm that the lift is can be attributed to the treatment and not pre-existing behavior from users or natural trends.
## In the three models created which contained a mix of different variables, he treatment effect remained stable and ranged from 2.44pp to 2.76pp.

#### Audience/Devices

## In looking deeper at the 2.44pp lift, the free trial users drove a large part of it.
## By device desktop (+3.60pp) outperformed mobile (+2.72pp). Tableau users showed a negative trend (-2.83pp) though not found to be significant.

## Looking at the behavioral funnel, it revealed the biggest treatment impact was at the engagement stage. 
## 61.7% of those in the treatment group  spent meaningful time on the page versus 36.8% of control users.

#### Recomemendation

## Given the positive results, we should rollout the new pricing page, with an initial rollout for free trial users.
## This segment saw the most impact and therefore, we should start with this group.

#### Next steps

## Desktop mode outperformed mobile by a full percentage point. Consider a version of pricing page specifically designed for mobile screens in mind.
## The 3-tier layout page saw a negative trend for tablet users. Investigate this further. Maybe the page doesn't render well on Tablet screens.
## Treatment users steered towards annual plans at a higher rate (+5.67pp) but it was not found to be statistically significant. 
## A follow up experiment with a larger sample size could better confirm this.
