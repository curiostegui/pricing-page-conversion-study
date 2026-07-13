## SaaS Pricing Page A/B Test & Causal Inference Conversion Study 

In this project, I'll be investigating whether the re-design of a SaaS companies pricing page will improve conversions. The new pricing page will replace a basic monthly/annual toggle with a 3-tier side-by-side layout feature clear plan comparisons. A 30-day randomized experiment was conducted using 5,000 users.

## Objective

The goal is to determine whether the redesigned pricing page will lead to greater improvements in the conversion rate, revenue, and annual subscriptions without increasing churn. 

## Approach

- Checked randomization across 6 pre-experiment variables using chi-square and t-tests.
- Performed A/B testing using chi-square and Mann-Whitney tests across primary and secondary metrics, along with power analysis to confirm adequate sample size.
- Used three causal inference methods - CUPED, Difference-in-Differences (DiD), and OLS regression - to verify the results of the A/B test.
- Conducted subgroup analysis breakdown across plan type, device, age, and country to see which user segment groups responded best to the treatment.
- Created a behavior funnel to see where the treatment had the most impact across the user journey.

## Key Insights

- The pricing page improved the conversion rate from 8.5% to 11.0%. It was also found to be statistically significant (p=0.0042).
- Revenue per user also improved by $ 0.25, and 30-day churn decreased by 1.87%.
- The conversion lift was primarily from the free trial users (+6.19%) - making them a priority segment for the rollout of the new page.
- Three different Causal Inference techniques found that the conversion was a result of the treatment and not because of pre-treatment user behavior, randomization imbalances, or natural trends.
- The largest treatment impact in the behavioral funnel occured around the engagement stage
- Desktop users had the highest conversion lift(+3.60), followed by mobile users (+2.72%). Tablet users showed a negative trend.


 ## Impact
 Delivered a recommendation plan that included the rollout of the redesigned pricing page, prioritizing free-trial users. In this plan, follow-up experiments and pain points were identified.
 
#### Keywords: Python, Tableau, A/B testing, Causal Inference, Difference-in-Differences, CUPED, OLS Regression, Statistical Testing, Chi-square Test, Mann-Whitney U, Power Analysis, Subgroup Analysis, Funnel Analysis, SaaS, Subscription Analysis, Product Analytics
 
 
