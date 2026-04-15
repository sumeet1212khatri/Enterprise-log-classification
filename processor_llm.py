
from __future__ import annotations
import os
import re
import time
import logging

from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN")
LLM_MODEL  = "mistralai/Mistral-7B-Instruct-v0.3"

VALID_CATEGORIES = ["Workflow Error", "Deprecation Warning"]

# Retry / timeout config
MAX_RETRIES     = 2
RETRY_DELAY_SEC = 1.0   # doubles on each retry (exponential backoff)
REQUEST_TIMEOUT = 5     # seconds — fail fast, do not hang pipeline

SYSTEM_PROMPT = (
    "You are an enterprise log classifier. "
    "Classify log messages into exactly one category. "
    "Return ONLY the category name — no explanation, no punctuation."
)

FEW_SHOT_EXAMPLES = [
    {
        "log":   "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.",
        "label": "Workflow Error",
    },
    {
        "log":   "The 'BulkEmailSender' feature is no longer supported. Use 'EmailCampaignManager' instead.",
        "label": "Deprecation Warning",
    },
    {
        "log":   "Invoice generation aborted for order ID 8910 due to invalid tax calculation module.",
        "label": "Workflow Error",
    },
]

# ── Prompt builder ───────────────────────────────────────────────────────────
def _build_messages(log_msg: str) -> list[dict]:
    categories_str = ", ".join(f'"{c}"' for c in VALID_CATEGORIES)
    user_content = (
        f'Classify the following log into one of these categories: {categories_str}.\n'
        'If none fits, return "Unclassified".\n\n'
    )
    for ex in FEW_SHOT_EXAMPLES:
        user_content += f'Log: {ex["log"]}\nCategory: {ex["label"]}\n\n'
    user_content += f"Log: {log_msg}\nCategory:"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ── Normalize raw LLM output ─────────────────────────────────────────────────
def _normalize(raw: str) -> str:
    """Map raw LLM output to a valid category or 'Unclassified'."""
    raw = raw.strip().strip('"').strip("'")
    for cat in VALID_CATEGORIES:
        if cat.lower() in raw.lower():
            return cat
    return "Unclassified"


# ── Main classify function ────────────────────────────────────────────────────
def classify_with_llm(log_msg: str) -> str:
    """
    Tier 3 LLM classifier with:
      - Timeout (REQUEST_TIMEOUT seconds)
      - Retry with exponential backoff (MAX_RETRIES attempts)
      - Explicit fallback to "Unclassified" on all error paths
    """
    # ── Inference with retry ─────────────────────────────────────────────────
    if not HF_TOKEN:
        logger.warning("[LLM] HF_TOKEN not set — returning Unclassified")
        return "Unclassified"

    from huggingface_hub import InferenceClient

    client  = InferenceClient(token=HF_TOKEN, timeout=REQUEST_TIMEOUT)
    delay   = RETRY_DELAY_SEC
    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 2):  # +2: initial + MAX_RETRIES
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=_build_messages(log_msg),
                max_tokens=15,
                temperature=0.1,
            )
            raw   = response.choices[0].message.content
            label = _normalize(raw)

            logger.debug(f"[LLM] Attempt {attempt}: '{raw.strip()}' → '{label}'")
            return label

        except Exception as e:
            # 🚨 JUGAD: Agar credits khatam hain (402), toh turant fallback do
            # Isse UI hang nahi hoga aur retry ka wait nahi karna padega
            if "402" in str(e) or "credits" in str(e).lower():
                logger.error(f"[LLM] Credits Finished (402). Returning Fallback Label.")
                return "Escalated: Manual Review Required (API Limit)"
            
            last_err = e
            if attempt <= MAX_RETRIES:
                logger.warning(f"[LLM] Attempt {attempt} failed ({e}), retrying in {delay:.1f}s…")
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                logger.error(f"[LLM] All attempts failed. Last error: {e}")

    return "Unclassified"


# ── Batch classify (serial — LLM is already rate-limited) ────────────────────
def classify_batch_llm(log_msgs: list[str]) -> list[str]:
    """Classify multiple logs through LLM. Each call is sequential to respect rate limits."""
    return [classify_with_llm(msg) for msg in log_msgs]


# ── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_logs = [
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.",
        "The 'ReportGenerator' module will be retired in version 4.0. Migrate to 'AdvancedAnalyticsSuite'.",
        "System reboot initiated by user 12345.",   # should be Unclassified
    ]
    for log in test_logs:
        result = classify_with_llm(log)
        print(f"{result:25s} | {log[:80]}")
