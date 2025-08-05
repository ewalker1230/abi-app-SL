from faker import Faker
import random
from datetime import date, timedelta

def generate_fake_campaign_brief():
    fake = Faker()
    
    # Campaign Overview
    campaign_name = fake.catch_phrase() # A catchy marketing phrase
    campaign_objective = fake.sentence(nb_words=10, variable_nb_words=True) + " " + fake.sentence(nb_words=8, variable_nb_words=True) 
    campaign_type = random.choice(["Brand Awareness", "Lead Generation", "Product Launch", "Sales Promotion", "Customer Retention"])
    start_date = fake.date_between(start_date="-1y", end_date="today")
    end_date = start_date + timedelta(days=random.randint(30, 180)) 
    
    # Target Audience
    target_audience_description = fake.paragraph(nb_sentences=3, variable_nb_sentences=True)
    target_demographics = {
        "Age Range": f"{random.randint(18, 25)}-{random.randint(35, 60)}",
        "Gender": random.choice(["All", "Male", "Female"]),
        "Location": fake.country(), # Generate a fake country
        "Interests": ", ".join(fake.words(nb=3))
    }
    
    # Messaging and Creative
    key_message = fake.sentence(nb_words=15, variable_nb_words=True)
    call_to_action = fake.sentence(nb_words=6, variable_nb_words=True)
    creative_direction = fake.paragraph(nb_sentences=2, variable_nb_sentences=True)
    
    # Budget and Metrics
    campaign_budget = f"${random.randint(5000, 500000):,}"
    success_metrics = {
        "Clicks": random.randint(1000, 100000),
        "Impressions": random.randint(50000, 5000000),
        "Conversion Rate": f"{round(random.uniform(0.5, 5.0), 2)}%",
        "Return on Ad Spend": f"{round(random.uniform(1.0, 5.0), 2)}x"
    }

    # Assembling the brief
    brief = {
        "Campaign Overview": {
            "Campaign Name": campaign_name,
            "Objective": campaign_objective,
            "Type": campaign_type,
            "Start Date": start_date.strftime("%Y-%m-%d"), 
            "End Date": end_date.strftime("%Y-%m-%d")
        },
        "Target Audience": {
            "Description": target_audience_description,
            "Demographics": target_demographics
        },
        "Messaging & Creative": {
            "Key Message": key_message,
            "Call to Action": call_to_action,
            "Creative Direction": creative_direction
        },
        "Budget & Metrics": {
            "Campaign Budget": campaign_budget,
            "Success Metrics": success_metrics
        }
    }
    return brief

# Generate a fake campaign brief
fake_brief = generate_fake_campaign_brief()

# Print to console
print("Generated Campaign Brief:")
for section, content in fake_brief.items():
    print(f"\n## {section}")
    for key, value in content.items():
        if isinstance(value, dict): # For nested dictionaries (like Demographics and Success Metrics)
            print(f"- {key}:")
            for sub_key, sub_value in value.items():
                print(f"  - {sub_key}: {sub_value}")
        else:
            print(f"- {key}: {value}")


# Save to file
filename = "my_output.txt"

with open(filename, 'w', encoding='utf-8') as file:
    file.write("Generated Campaign Brief:\n")
    for section, content in fake_brief.items():
        file.write(f"\n## {section}\n")
        for key, value in content.items():
            if isinstance(value, dict):  # For nested dictionaries
                file.write(f"- {key}:\n")
                for sub_key, sub_value in value.items():
                    file.write(f"  - {sub_key}: {sub_value}\n")
            else:
                file.write(f"- {key}: {value}\n")

print(f"\nCampaign brief saved to {filename}")