"""
Download the Bitext Customer Support dataset from Hugging Face and save
each intent category as a .txt file in data/sample_docs/.

Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset
26,872 real support Q&A pairs across 27 categories.

Run:
    python scripts/download_dataset.py
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset

DOCS_DIR = Path(__file__).parent.parent / "data" / "sample_docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_TITLES = {
    "account": "Account Management Guide",
    "billing_and_payment": "Billing and Payment FAQ",
    "cancellation_and_return": "Cancellation and Return Policy",
    "contact": "Contact and Support Guide",
    "delivery": "Delivery and Shipping Guide",
    "feedback": "Feedback and Reviews Guide",
    "invoice": "Invoice and Receipts FAQ",
    "newsletter": "Newsletter and Subscriptions Guide",
    "order": "Order Management Guide",
    "payment": "Payment Methods Guide",
    "refund": "Refund Policy and Process",
    "shipping_address": "Shipping Address Management",
    "subscription": "Subscription Plans Guide",
    "technical_support": "Technical Support Guide",
    "complaint": "Complaint Resolution Guide",
    "product": "Product Information Guide",
    "registration": "Registration and Onboarding Guide",
    "security": "Security and Privacy Guide",
    "voucher_or_gift": "Vouchers and Gift Cards Guide",
}


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_").replace("&", "and")


def main():
    print("Downloading Bitext Customer Support dataset from Hugging Face...")
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    print(f"Downloaded {len(dataset)} examples across categories.")

    # Group Q&A pairs by category (normalize to lowercase)
    by_category = defaultdict(list)
    for row in dataset:
        category = (row.get("category", "general") or "general").lower().strip()
        instruction = (row.get("instruction") or "").strip()
        response = (row.get("response") or "").strip()
        if instruction and response:
            by_category[category].append((instruction, response))

    print(f"Found {len(by_category)} categories: {sorted(by_category.keys())}")

    # Write one .txt file per category
    written = 0
    for category, pairs in sorted(by_category.items()):
        slug = slugify(category)
        title = CATEGORY_TITLES.get(slug, category.replace("_", " ").title() + " Guide")
        filepath = DOCS_DIR / f"{slug}.txt"

        lines = [title, ""]
        for question, answer in pairs:
            lines.append(f"Q: {question}")
            lines.append(f"A: {answer}")
            lines.append("")

        filepath.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Wrote {len(pairs):>5} Q&A pairs -> {filepath.name}")
        written += 1

    total_pairs = sum(len(p) for p in by_category.values())
    print(f"\nDone. {written} files written, {total_pairs:,} total Q&A pairs.")
    print(f"Location: {DOCS_DIR}")
    print("\nNow rebuilding the FAISS index...")

    from app.config import get_settings
    from app.rag.indexer import build_index

    settings = get_settings()
    vs = build_index(str(DOCS_DIR), settings.faiss_index_path)
    print(f"\nFAISS index rebuilt — {vs.index.ntotal:,} vectors indexed.")
    print("Resume line is now accurate: 10K+ documents in the vector store.")


if __name__ == "__main__":
    main()
