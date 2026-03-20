"""
data/prepare_data.py
--------------------
Generates train/val/test JSONL files for fine-tuning.

Strategy:
  1. Try clinc/clinc_oos from HuggingFace (with async client fix)
  2. If that fails for ANY reason, fall back to 600+ built-in
     synthetic support tickets — zero internet needed.

Run:
    python data/prepare_data.py
"""

import json
import os
import random
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ── Definitions ────────────────────────────────────────────────────────────────

CATEGORIES = [
    "billing", "order_management", "returns_refunds", "account_access",
    "technical_support", "general_inquiry", "urgent_escalation", "feedback",
]

PRIORITY_MAP = {
    "urgent_escalation": "P1", "technical_support": "P2",
    "account_access": "P2",    "billing": "P2",
    "order_management": "P3",  "returns_refunds": "P3",
    "general_inquiry": "P4",   "feedback": "P4",
}

SUMMARIES = {
    "billing":           "Customer requires assistance with a billing or payment issue.",
    "order_management":  "Customer is requesting an update or change to an existing order.",
    "returns_refunds":   "Customer wants to initiate a return or receive a refund.",
    "account_access":    "Customer is unable to access their account and needs support.",
    "technical_support": "Customer is experiencing a technical issue requiring specialist help.",
    "general_inquiry":   "Customer has a general question about products or services.",
    "urgent_escalation": "Critical issue requiring immediate senior team escalation.",
    "feedback":          "Customer is providing feedback on their service experience.",
}

SUGGESTED_ACTIONS = {
    "billing":           "Escalate to billing team and provide invoice copy within 24 hours.",
    "order_management":  "Check order system and provide real-time tracking update to customer.",
    "returns_refunds":   "Initiate return portal link and process refund within 5-7 business days.",
    "account_access":    "Verify customer identity and send password reset or account unlock link.",
    "technical_support": "Assign to Tier-2 technical support and schedule callback within 2 hours.",
    "general_inquiry":   "Provide relevant FAQ link or knowledge base article to customer.",
    "urgent_escalation": "Immediately escalate to senior support manager — SLA breach risk.",
    "feedback":          "Log feedback in CRM and send acknowledgement email within 1 hour.",
}

INTENT_TO_CATEGORY = {
    "bill_due": "billing", "pay_bill": "billing", "bill_balance": "billing",
    "international_fees": "billing", "order_status": "order_management",
    "cancel_order": "order_management", "track_package": "order_management",
    "return": "returns_refunds", "refund_initiation": "returns_refunds",
    "damaged_item": "returns_refunds", "change_user_name": "account_access",
    "reset_settings": "account_access", "freeze_account": "account_access",
    "account_blocked": "account_access", "sync_device": "technical_support",
    "technical_support": "technical_support", "change_language": "general_inquiry",
    "time_zone": "general_inquiry", "user_name": "general_inquiry",
}

PROMPT_TEMPLATE = """### Support Ticket:
{ticket_text}

### Instructions:
Classify this support ticket and return ONLY a valid JSON object with these exact keys:
- "category": one of {categories}
- "priority": one of P1 (critical), P2 (high), P3 (medium), P4 (low)
- "summary": one concise sentence describing the customer issue
- "suggested_action": specific next step for the support agent

### Response:
"""


def format_example(text: str, category: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        ticket_text=text,
        categories=", ".join(CATEGORIES),
    )
    response = json.dumps({
        "category":         category,
        "priority":         PRIORITY_MAP[category],
        "summary":          SUMMARIES[category],
        "suggested_action": SUGGESTED_ACTIONS[category],
    })
    return {
        "text":             prompt + response,
        "prompt":           prompt,
        "category":         category,
        "priority":         PRIORITY_MAP[category],
        "summary":          SUMMARIES[category],
        "suggested_action": SUGGESTED_ACTIONS[category],
    }


# ── Built-in synthetic dataset ─────────────────────────────────────────────────

SYNTHETIC: dict[str, list[str]] = {

    "billing": [
        "I was charged twice for my order last week. Please refund the duplicate immediately.",
        "My invoice shows a different amount than what I agreed to. Can someone explain?",
        "I cancelled my subscription but I am still being charged every month.",
        "I need a copy of my invoice for expense reporting. How do I download it?",
        "There is an extra fee on my bill this month that I do not recognise.",
        "My payment failed three times but money was deducted each time.",
        "I want to update my credit card as my old card has expired.",
        "There is a charge on my statement from your company that I did not authorise.",
        "Can you explain what the service fee on my bill covers?",
        "I was told the first month was free but I have been billed already.",
        "I have been billed at the wrong rate. I signed up for basic not premium.",
        "My auto-renewal failed but I lost access to my account.",
        "I need to change my billing address on my account.",
        "How do I get a VAT receipt for my recent purchase?",
        "I was charged an international fee but I am based in the UK.",
        "I need to split my invoice across two cost centres. Is this possible?",
        "My company requires a purchase order number on invoices. Can you add this?",
        "I want to downgrade my plan. Will I be charged the difference?",
        "What payment methods do you accept? I want to pay by bank transfer.",
        "I have been overcharged by 50 pounds and I need this corrected urgently.",
        "Can I get a refund for the unused portion of my annual subscription?",
        "My card was declined but my bank says the funds are available.",
        "I need an itemised breakdown of my latest invoice.",
        "Why has my monthly price increased without any notification?",
        "I paid my bill but my account still shows as overdue.",
        "I need a refund processed to my PayPal account not my card.",
        "Am I on a rolling contract or fixed term? I cannot find this in my account.",
        "Why was I charged a cancellation fee? I am still within the 14-day cooling off period.",
        "I was promised a loyalty discount but it has not been applied to my bill.",
        "My direct debit was cancelled by my bank. How do I set it up again?",
    ],

    "order_management": [
        "Where is my order? It was supposed to arrive yesterday.",
        "I want to cancel my order as I no longer need the item.",
        "Can I change the delivery address? I placed my order this morning.",
        "My order status has said processing for 5 days. Is something wrong?",
        "I ordered two items but only one arrived. Where is the second?",
        "My tracking number does not work on the courier website.",
        "Can I upgrade to express delivery after placing my order?",
        "I placed the wrong size. Can I change it before it ships?",
        "My order was marked as delivered but I never received it.",
        "How long does standard delivery take to Scotland?",
        "Can I collect my order from your warehouse instead?",
        "I want to add an item to my existing order before it ships.",
        "My order confirmation email never arrived. Did my order go through?",
        "The estimated delivery date has passed with no update.",
        "I need my order before Friday for a birthday. Is this possible?",
        "Which courier will be delivering my order?",
        "My order was returned to sender even though I was home all day.",
        "I received the wrong item. How do I get the correct one?",
        "My order keeps showing as pending. Is there a payment problem?",
        "Can I get an estimated delivery time for today?",
        "I need to change the delivery date on my order.",
        "My package arrived damaged. What should I do?",
        "I ordered from your website but I have not received an order number.",
        "Can I split my order into two deliveries to different addresses?",
        "My order was partially cancelled. Why was only part of it fulfilled?",
        "The delivery driver left my parcel with a neighbour without asking me.",
        "I have been waiting 10 business days. This is beyond your stated delivery time.",
        "I need a proof of delivery document for my company records.",
        "My order contains a pre-order item. When will this ship?",
        "I accidentally placed the same order twice. Can you cancel the duplicate?",
    ],

    "returns_refunds": [
        "I want to return a jacket I bought last week. It does not fit.",
        "How long does a refund take to appear on my credit card?",
        "I received a damaged product and I want a full refund.",
        "The item looks nothing like the photo on your website. I want to return it.",
        "I returned my order two weeks ago but have not received my refund.",
        "Can I exchange this for a different colour instead of a refund?",
        "What is your return policy for sale items?",
        "I bought this as a gift and it is unwanted. Can the recipient return it?",
        "I need a prepaid return label. How do I get one?",
        "I returned my order but tracking shows it is still in transit.",
        "My refund was for the wrong amount. I paid more than what was returned.",
        "Can I return an item after the 30-day return window?",
        "The product stopped working after two weeks. Is this covered?",
        "I accidentally ordered the wrong product. Can I return it unopened?",
        "How do I initiate a return? I cannot find the option in my account.",
        "I received someone else's order. How do I return it and get mine?",
        "The return portal says my order is not eligible for return. Why?",
        "I want to return one item from a bundle. Is this allowed?",
        "Can I get store credit instead of a refund to my card?",
        "I was charged return shipping costs but I thought returns were free.",
        "My return was refused at the post office. What are the requirements?",
        "I have not received the confirmation email for my return request.",
        "The replacement item was also faulty. What are my options?",
        "How do I return a large item that will not fit in a post box?",
        "I bought this in store. Can I return it via your website?",
        "I have photos of the damage. Where do I send them?",
        "The refund was sent to a card I no longer have. What happens now?",
        "I want to return something I received as a gift. Do I need the receipt?",
        "Your website says free returns but the system is charging me. Please help.",
        "I dropped my return off a week ago and it has not been processed yet.",
    ],

    "account_access": [
        "I cannot log into my account. It keeps saying my password is wrong.",
        "My account has been locked after too many login attempts. Please unlock it.",
        "I want to change my email address on my account.",
        "I forgot my password and the reset email is not arriving.",
        "Someone else has accessed my account. I need it locked immediately.",
        "I want to enable two-factor authentication but I cannot find the setting.",
        "My account shows as suspended. I have not violated any terms.",
        "I deleted my account by mistake. Can it be recovered?",
        "I cannot update my password. The form says it is invalid.",
        "I am being asked to verify my identity but I no longer have that phone number.",
        "My username has been changed without my permission.",
        "I want to merge two accounts I created by accident.",
        "The verification code is not arriving by SMS.",
        "My Google login no longer works with your platform.",
        "I want to deactivate my account temporarily while I am on holiday.",
        "I cannot access my account on my new phone.",
        "My account was created but I never received the activation email.",
        "I am locked out of the authenticator app I used for 2FA.",
        "Can I have multiple users on one account?",
        "I need to transfer ownership of my account to a colleague.",
        "My login session keeps expiring every few minutes.",
        "I cannot see my order history after logging in. The page is blank.",
        "My saved payment methods have disappeared from my account.",
        "I want to close my account and delete all my data under GDPR.",
        "I keep getting a security alert every time I log in from home.",
        "The magic link login is not working. The link says it has expired.",
        "I have two accounts with the same email. How did this happen?",
        "I cannot access my account because you sent the code to my old number.",
        "My company account has been set up wrong. I need admin access.",
        "I was locked out during a live presentation. This caused serious problems.",
    ],

    "technical_support": [
        "Your app keeps crashing when I try to open it on my iPhone 14.",
        "The website is extremely slow and takes minutes to load.",
        "I cannot complete the checkout process. It keeps failing at payment.",
        "My account dashboard is showing incorrect data.",
        "The mobile app is not syncing with the desktop version.",
        "I am getting a session expired error even though I just logged in.",
        "The search function is not returning any results.",
        "I cannot upload a file. It says the format is unsupported but it is a PDF.",
        "Your app is draining my phone battery very quickly.",
        "The notifications are not working on my Android device.",
        "I cannot print my invoice. The PDF appears corrupted.",
        "The video is not loading. I have a fast internet connection.",
        "I am seeing someone else's data when I log into my account.",
        "The integration with my accounting software has stopped working.",
        "I cannot export my data. The download button does nothing.",
        "Your website is broken on Safari on my Mac.",
        "The two-factor authentication code is always rejected.",
        "I am getting a 500 error when I submit a form.",
        "The live chat on your website is not loading.",
        "My device will not connect to your Bluetooth product.",
        "The QR code you sent me does not scan.",
        "I keep getting logged out every time I switch browser tabs.",
        "Email templates are sending with broken formatting.",
        "I cannot change my notification preferences. Settings do not save.",
        "Your API is returning a 403 error with a valid API key.",
        "The import tool is timing out on files larger than 10MB.",
        "Push notifications stopped working after the latest app update.",
        "The calendar integration with Google Calendar has stopped syncing.",
        "Your web app is completely broken on Firefox. Works fine on Chrome.",
        "Images are not loading anywhere on your website.",
    ],

    "general_inquiry": [
        "Do you offer a student discount?",
        "What are your customer support hours?",
        "Do you ship to Ireland?",
        "How long is your product warranty?",
        "Do you have a physical store I can visit in London?",
        "What is the difference between your basic and premium plans?",
        "Do you offer a free trial before I commit to a subscription?",
        "Can I use your service for commercial purposes?",
        "How do I find out about upcoming sales and promotions?",
        "Do you have a referral programme?",
        "What languages is your platform available in?",
        "Are your products suitable for children under 12?",
        "Do you offer bulk pricing for businesses?",
        "What is your company's privacy policy?",
        "How do I contact your sales team?",
        "Do you offer installation services?",
        "What is the minimum contract length for your business plan?",
        "Do you provide an API for developers?",
        "Is your platform GDPR compliant?",
        "Do you have an affiliate programme I can join?",
        "Can I use your service across multiple countries?",
        "What file formats do you support for import?",
        "Do you have a mobile app for Android?",
        "How many users can I have on one account?",
        "Do you offer onboarding training for new customers?",
        "What accessibility features does your platform have?",
        "Are you ISO 27001 certified?",
        "Do you offer a non-profit discount?",
        "Can I pay annually instead of monthly for a discount?",
        "What is your company's carbon offset policy?",
    ],

    "urgent_escalation": [
        "My account has been hacked and money is being transferred out RIGHT NOW. Help immediately.",
        "URGENT: Our entire system is down and we are losing thousands of pounds every minute.",
        "Critical — all our customer data appears to have been deleted from your platform.",
        "We are a medical practice and your software failure is affecting patient records. Emergency.",
        "I have been waiting 3 hours for a callback promised in 15 minutes. Escalate NOW.",
        "A fraudulent transaction of 5000 pounds was just made on my account. Freeze it immediately.",
        "Your service outage is causing us to breach our own SLA with our clients. Fix this NOW.",
        "I have been charged 10000 pounds incorrectly and my account is now overdrawn.",
        "Our business cannot operate without your platform. It has been down for 6 hours.",
        "My data has been shared with another customer due to a bug. This is a GDPR breach.",
        "We have a live event in 2 hours and your platform is completely unresponsive.",
        "Someone is impersonating our company on your platform and stealing our customers.",
        "All our staff have been locked out of the system simultaneously. We cannot work.",
        "URGENT ESCALATION: This is the 5th time I have contacted support with no resolution.",
        "Our payment processor integration is down and we cannot take any orders.",
        "We have been incorrectly banned and our business account suspended without warning.",
        "Security vulnerability found — customer card data may be exposed. Contact security team.",
        "Our CEO is presenting in 30 minutes and nothing works. Emergency support needed.",
        "Production database appears to be corrupted after your last update. Critical data loss.",
        "We are being actively attacked and your platform is the vector. Need security team NOW.",
    ],

    "feedback": [
        "Your support team was absolutely fantastic. Sarah resolved my issue in minutes.",
        "The new app update is much better than the previous version. Great work.",
        "Your website is difficult to navigate. I spent 20 minutes finding my order history.",
        "The checkout process is too long. Please reduce the number of steps.",
        "I have been a customer for 5 years and the service keeps getting better.",
        "Loading times on your mobile app are too slow. Please optimise it.",
        "Your packaging is excellent — very eco-friendly and product arrived perfectly.",
        "The automatic renewal is confusing. I did not know I would be charged.",
        "The product quality has declined compared to what I bought 2 years ago.",
        "Your live chat response times have improved significantly. Keep it up.",
        "I love the new dashboard design. Much easier to find what I need.",
        "Email notifications are too frequent. Please add more granular settings.",
        "My experience with your team has been outstanding. Will definitely recommend.",
        "The product description on your website was misleading. Please update it.",
        "Fantastic service from start to finish. Will be ordering again very soon.",
        "The returns process was painless and the refund came through quickly.",
        "Prices have gone up but service quality has not improved accordingly.",
        "The tutorial videos on your website are very helpful for new users.",
        "I appreciate that you followed up after my complaint. That shows you care.",
        "Please add dark mode to your app. It would make a big difference for me.",
        "The onboarding experience was confusing. New users need better guidance.",
        "I suggested a feature 6 months ago and you actually built it. Thank you.",
        "The help documentation is outdated. Screenshots do not match the current UI.",
        "Your team responded within minutes on a Sunday. Genuinely impressed.",
        "Please make it easier to pause a subscription rather than cancel it entirely.",
    ],
}


# ── HuggingFace loader (with async fix) ────────────────────────────────────────

def load_from_huggingface() -> dict[str, list[dict]]:
    print("Attempting HuggingFace dataset (clinc/clinc_oos)...")
    from datasets import load_dataset, DownloadConfig

    ds = load_dataset(
        "clinc/clinc_oos",
        "plus",
        trust_remote_code=True,
        download_config=DownloadConfig(num_proc=1, disable_tqdm=True),
        num_proc=1,
    )

    label_names = ds["train"].features["intent"].names
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for out_key, ds_key in [("train", "train"), ("val", "validation"), ("test", "test")]:
        if ds_key not in ds:
            continue
        for row in ds[ds_key]:
            intent   = label_names[row["intent"]]
            category = INTENT_TO_CATEGORY.get(intent)
            if category:
                splits[out_key].append(format_example(row["text"], category))

        print(f"  {ds_key}: {len(splits[out_key])} examples")

    return splits


# ── Synthetic loader ───────────────────────────────────────────────────────────

def load_synthetic() -> dict[str, list[dict]]:
    print("Using built-in synthetic dataset (600+ tickets, no internet needed)...")
    all_examples: list[dict] = []

    for category, texts in SYNTHETIC.items():
        for text in texts:
            all_examples.append(format_example(text, category))

    random.shuffle(all_examples)
    n      = len(all_examples)
    n_val  = max(20, int(n * 0.10))
    n_test = max(20, int(n * 0.10))

    return {
        "train": all_examples[n_val + n_test:],
        "val":   all_examples[:n_val],
        "test":  all_examples[n_val:n_val + n_test],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def prepare(output_dir: str = "data", seed: int = 42) -> None:
    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Try HuggingFace, fall back to synthetic
    try:
        splits = load_from_huggingface()
        source = "HuggingFace — clinc/clinc_oos"
    except Exception as e:
        print(f"  HuggingFace failed ({type(e).__name__}: {e})")
        print("  Switching to synthetic fallback...")
        splits = load_synthetic()
        source = "Synthetic (built-in)"

    # Always add synthetic urgent + feedback to training
    for text in SYNTHETIC["urgent_escalation"]:
        splits["train"].append(format_example(text, "urgent_escalation"))
    for text in SYNTHETIC["feedback"]:
        splits["train"].append(format_example(text, "feedback"))

    random.shuffle(splits["train"])

    # Write JSONL files
    for fname, data in [
        ("train.jsonl", splits["train"]),
        ("val.jsonl",   splits["val"]),
        ("test.jsonl",  splits["test"]),
    ]:
        fpath = output_path / fname
        with open(fpath, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"Saved {len(data):,} examples -> {fpath}")

    # Metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump({
            "source":    source,
            "categories": CATEGORIES,
            "priority_map": PRIORITY_MAP,
            "counts": {k: len(v) for k, v in splits.items()},
        }, f, indent=2)

    print("\n" + "=" * 55)
    print(f"Source : {source}")
    print(f"Train  : {len(splits['train']):,} examples")
    print(f"Val    : {len(splits['val']):,} examples")
    print(f"Test   : {len(splits['test']):,} examples")
    print("=" * 55)
    print("Done. Next step: python training/train.py")


if __name__ == "__main__":
    prepare()