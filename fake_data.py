from faker import Faker
import random
import pandas as pd

# Initialize Faker
fake = Faker()

# Set a seed for reproducibility (optional)
Faker.seed(42)

def generate_fake_ad_campaign_data(num_campaigns=100):
    """Generates a list of dictionaries representing fake ad campaign data."""
    ad_campaign_data = []
    ad_platforms = ["Facebook", "Google Ads", "Instagram", "LinkedIn", "Twitter", "TikTok", "YouTube", "Snapchat"]
    ad_types = ["Display", "Video", "Search", "Social", "Retargeting", "Influencer", "Native", "Banner"]
    campaign_statuses = ["Draft", "Active", "Paused", "Completed", "Scheduled"]
    targeting_methods = ["Demographic", "Interest-based", "Behavioral", "Lookalike", "Custom Audience", "Geographic"]

    for i in range(num_campaigns):
        campaign_name = fake.catch_phrase() + " " + fake.word().capitalize() + " Ad Campaign"
        start_date = fake.date_between(start_date='-1y', end_date='today')
        end_date = fake.date_between(start_date=start_date, end_date='+6m')
        budget = round(random.uniform(500, 25000), 2)
        daily_budget = round(budget / random.randint(7, 30), 2)
        platform = random.choice(ad_platforms)
        ad_type = random.choice(ad_types)
        status = random.choice(campaign_statuses)
        
        # Generate realistic ad metrics
        impressions = random.randint(1000, 100000) if status in ["Active", "Completed"] else random.randint(0, 1000)
        clicks = random.randint(50, 5000) if status in ["Active", "Completed"] else random.randint(0, 100)
        ctr = round(clicks / impressions, 4) if impressions > 0 else 0.0
        cpc = round(random.uniform(0.50, 5.00), 2) if clicks > 0 else 0.0
        conversions = random.randint(5, 500) if status in ["Active", "Completed"] else random.randint(0, 10)
        conversion_rate = round(conversions / clicks, 4) if clicks > 0 else 0.0
        cost_per_conversion = round((cpc * clicks) / conversions, 2) if conversions > 0 else 0.0
        
        targeting = random.choice(targeting_methods)
        age_range = random.choice(["18-24", "25-34", "35-44", "45-54", "55+", "18-34", "25-54"])
        gender_targeting = random.choice(["All", "Male", "Female", "Non-binary"])
        
        ad_campaign = {
            "campaign_id": i + 1,
            "campaign_name": campaign_name,
            "platform": platform,
            "ad_type": ad_type,
            "start_date": start_date,
            "end_date": end_date,
            "status": status,
            "total_budget": budget,
            "daily_budget": daily_budget,
            "impressions": impressions,
            "clicks": clicks,
            "ctr": ctr,
            "cpc": cpc,
            "conversions": conversions,
            "conversion_rate": conversion_rate,
            "cost_per_conversion": cost_per_conversion,
            "targeting_method": targeting,
            "age_range": age_range,
            "gender_targeting": gender_targeting,
            "total_spent": round(cpc * clicks, 2),
            "roas": round(random.uniform(1.5, 8.0), 2) if conversions > 0 else 0.0
        }
        ad_campaign_data.append(ad_campaign)
    return ad_campaign_data

# Generate 100 fake ad campaigns
fake_ad_campaigns = generate_fake_ad_campaign_data(num_campaigns=100)

# Convert to a Pandas DataFrame for easier analysis and storage
df_ad_campaigns = pd.DataFrame(fake_ad_campaigns)

# Save to CSV file
df_ad_campaigns.to_csv("fake_ad_campaigns.csv", index=False)
print("Generated fake ad campaign data with", len(df_ad_campaigns), "campaigns")
print("Columns:", list(df_ad_campaigns.columns))
print("\nFirst few rows:")
print(df_ad_campaigns.head())