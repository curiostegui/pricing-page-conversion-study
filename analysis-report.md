
## Introduction

Subscription SaaS (Software-as-a-Service) businesses rely heavily on their pricing page to keep current users and attract new customers. It is an important touchpoint in the customer journey. A poorly set up pricing page can lead to disinterest and cause users to abandon the website, even if they are interested in the product.

In this hypothetical scenario, a SaaS company is looking to revamp its payment plan. Currently, the pricing page uses a monthly/annual toggle format that presents two plan options with minimal context.

It presents little information on the value of higher-tier payment plans for prospective customers. As a result, the conversion rates for free trial users have plateaued. If there were a more robust payment page, the pricing experience for users could improve.

## Experiment

A redesigned pricing page 3-tier plan with plan options side by side with supporting information was developed and tested.
The experiment ran for 30 days from January 15 to February 14, 2024. 

5000 users were randomly assigned and evenly split into two groups:

- Control group (2500 users): Saw the prior monthly/annual toggle pricing page. 

- Treatment group (2500 users): Saw the new 3-tier pricing page.

The primary objective is to test whether the redesigned pricing page increases the rate at which users convert from free to a paid subscription.

Outside of the main conversion metric. I tracked and identified both secondary and guardrail metrics to 
ensure that improvements in conversion did not negatively affect other outcomes:

- Revenue per User: Did the treatment generate more revenue per user exposed in the experiment?
- 30-day churn rate: Did treatment users stay, or did they cancel their plan?
- Annual plan: Did the new page shift converters towards annual plans?
  
## Methodology

**Data Acquisition**

Given the difficulty in finding real company SaaS user behavioral data, I generated a synthetic dataset using Claude (Anthropic). In collaboration, I tried to simulate real SaaS subscription data, which included user demographics, pricing page interactions, and conversion results.

**Statistical Methods**

The analysis will combine statistical testing with causal inference methods to evaluate the experiment.

**A/B Testing:** Classical hypothesis test that tests whether variation in behaviors on a website is statistically significant. Power analysis will be performed first to confirm that our sample size is large enough to detect a meaningful effect. Both primary and secondary metrics will be tested using the appropriate statistical test. Results will be reported with p-values and lift figures.

**Causal Inference:** To complement the results of the A/B test, causal inference techniques will be used to determine the confidence in the results:
- CUPED (Controlled-experiment Using Pre-Experiment Data) was used to test whether pre-experiment user behavior could reduce variation in the treatment effect estimate.
- Difference-in-Differences (DiD) was used to measure whether the treatment made meaningful user engagement changes outside of natural trends by seeing how groups changed from pre-experiment to post-experiment.
- OLS Regression looked at the treatment effect while also pairing it with multiple user-level characteristics to see the stability of the treatment coefficient across different regression models.

**Other Studies to be Performed**

**Subgroup Analysis:** I'll be examining the conversion lift broken down across four segments: plan type, device type, age group and country. Interactions test will then be used to confirm whether the difference across segments was statistically significant.

**Behavioral Funnel:** A customer behavior funnel was created to see where in the user journey the treatment had the most impact. Each stage was analyzed seperately for both control and treatment groups. Using overall funnel rates and step-over-step conversion rates, I looked at what the overall pipeline health is. 

## Data Overview

There are 5,000 rows and 18 columns that are comprised of a mix of identifiers (user_id), experiment assignment (experiment_group, experiment_start_date), 
user background (signup_date, country, device_type, etc.), pre-experiment behavior (engagement_score_pre, sessions_pre_30d, etc.), pricing page behavior (pricing_page_views, time_on_pricing_page_sec) and post outcome (converted_to_paid, plan_chosen, etc.) data.

# Table 1: Variable Names and Definitions

| Variable Name | Definition |
|---|---|
| `user_id` | Unique identifier for each user |
| `experiment_group` | Pricing page version the user was shown (control or treatment) |
| `signup_date` | Date the user first signed up for the platform |
| `experiment_start_date` | Date the experiment began (January 15, 2024 for all users) |
| `country` | Country where the user is located |
| `device_type` | Primary device used to access the pricing page (mobile, desktop, tablet) |
| `age_bucket` | User's age group (18-24, 25-34, 35-44, 45-54, 55+) |
| `plan_at_experiment_start` | Subscription plan the user held when the experiment began (free trial, monthly, annual) |
| `engagement_score_pre` | Composite score (1–10) measuring how actively the user engaged with the platform before the experiment |
| `sessions_pre_30d` | Number of platform sessions in the 30 days before the experiment started |
| `pricing_page_views` | Number of times the user visited the pricing page during the experiment |
| `time_on_pricing_page_sec` | Total time in seconds the user spent on the pricing page during the experiment |
| `converted_to_paid` | Whether the user upgraded to a paid plan during the 30-day window (1 = yes, 0 = no) |
| `plan_chosen` | Plan selected at conversion (monthly, annual, or none if no conversion) |
| `days_to_conversion` | Number of days between experiment start and conversion (blank if user did not convert) |
| `revenue_30d_usd` | Revenue generated by the user in the 30 days following experiment start |
| `sessions_post_30d` | Number of platform sessions in the 30 days after the experiment started |
| `churned_within_30d` | Whether an existing paid user cancelled their subscription within 30 days (1 = yes, 0 = no) |
