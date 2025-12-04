# dataset_gen.py
import pandas as pd
import random

# ---------------------------
# Word Pools
# ---------------------------
spam_words = [
    "free", "offer", "win", "prize", "cash", "lottery", "discount", "urgent",
    "winner", "click", "buy", "order", "money", "limited", "bonus", "credit",
    "loan", "deal", "cheap", "lowest", "amazing", "exclusive", "jackpot",
    "claim", "earn", "guarantee", "lifetime", "payout"
]

ham_words = [
    "meeting", "project", "schedule", "family", "dinner", "office", "report",
    "team", "update", "payment", "invoice", "task", "discussion", "confirm",
    "training", "developer", "deadline", "feedback", "performance", "document",
    "attendance", "agenda", "client", "budget", "review", "minutes"
]

links = [
    "http://promo-offer.com", "https://win-big.net", "https://secure-payment.com",
    "https://update-info.org", "http://official-site.com", "https://claim-reward.net",
    "https://limited-deal.info"
]

names = ["John", "David", "Priya", "Anita", "Sam", "Kiran", "Riya", "Rohan", "Sneha", "Vikram", "Meera"]

# Function to generate a sentence
def generate_sentence(words, min_words=8, max_words=20):
    return " ".join(random.choices(words, k=random.randint(min_words, max_words)))

# ---------------------------
# Generate Dataset
# ---------------------------
data = []
total = 10000  # 10,000 emails
half = total // 2

# Generate Spam
for _ in range(half):
    text = (
        f"Hello {random.choice(names)}, "
        f"You have a {random.choice(['new', 'exclusive', 'special'])} "
        f"{random.choice(['offer', 'reward', 'deal'])}! "
        + generate_sentence(spam_words)
        + f" Click here: {random.choice(links)} "
        + f"Valid till {random.randint(1, 30)}/12/2025. "
        + f"Use code {random.randint(10000, 99999)} NOW!!!"
    )
    data.append([text, "spam"])

# Generate HAM (improved with greetings)
for _ in range(half):
    text = (
        f"Hello {random.choice(names)},\n"
        f"I hope you are doing well. "
        f"This email is regarding your upcoming {random.choice(['project', 'meeting', 'schedule','invoice','presentation','document'])}. "
        + generate_sentence(ham_words)
        + f". Please confirm the update by {random.randint(1, 30)}/12/2025. "
        + random.choice(["Thanks & Regards.", "Warm Regards.", "Sincerely.", "Best Wishes."])
    )
    data.append([text, "ham"])

# Shuffle
random.shuffle(data)

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Email_Text", "Label"])

# Save CSV
df.to_csv("email_dataset_2.csv", index=False)
print("Generated dataset saved -> email_dataset.csv")
print(df.head())
