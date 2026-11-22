"""Generate a reproducible card-transaction dataset with ~0.6-0.8% fraud.

The sandbox could not download a public fraud dataset, so this generator produces data with the
same shape as a card-payment feed: one row per transaction, a binary label, skewed amounts, risky
merchant categories, international and new-device flags, missing values and a 180-day timeline.
The fraud label depends on those attributes through a logistic model, so the learned signal is
realistic but the absolute metrics will differ on real data. Swap in a real CSV with the same
columns at data/raw/transactions.csv and nothing else changes.
"""
import argparse

import numpy as np
import pandas as pd

from src import config
from src.config import HIGH_RISK_CATEGORIES, N_DAYS, START_DATE, RANDOM_STATE


INTERCEPT = -7.4  # calibrated so the overall fraud rate is about 0.6%


def generate(n_cards: int = 5000, mean_txns_per_card: float = 60.0, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(START_DATE)

    activity = rng.gamma(2.0, 0.5, n_cards)
    counts = rng.poisson(mean_txns_per_card * activity / activity.mean())
    card_idx = np.repeat(np.arange(n_cards), counts)
    card_age0 = rng.integers(1, 3000, n_cards)
    seconds = rng.uniform(0, N_DAYS * 86400, len(card_idx)).astype(np.int64)

    # Compromised cards: ~3% of cards get a burst of 25 extra transactions within 2 days.
    # This is the scenario velocity features are meant to catch.
    compromised = rng.random(n_cards) < 0.03
    burst_cards = np.repeat(np.where(compromised)[0], 25)
    burst_start = rng.uniform(0, (N_DAYS - 2) * 86400, n_cards)
    burst_secs = (burst_start[burst_cards] + rng.uniform(0, 2 * 86400, len(burst_cards))).astype(np.int64)
    is_burst = np.concatenate([np.zeros(len(card_idx), bool), np.ones(len(burst_cards), bool)])
    card_idx = np.concatenate([card_idx, burst_cards])
    seconds = np.concatenate([seconds, burst_secs])
    n = len(card_idx)
    start = pd.Timestamp(START_DATE)
    timestamp = start + pd.to_timedelta(seconds, unit="s")
    hour = timestamp.hour

    cats = config.CATEGORY_VALUES["merchant_category"]
    cat_p = np.array([0.22, 0.10, 0.14, 0.10, 0.08, 0.07, 0.04, 0.05, 0.03, 0.17])
    merchant = rng.choice(cats, n, p=cat_p / cat_p.sum())
    channel = rng.choice(["online", "pos", "atm"], n, p=[0.45, 0.45, 0.10])
    channel[is_burst & (rng.random(n) < 0.85)] = "online"

    amount_mult = np.ones(n)
    amount_mult[np.isin(merchant, ["luxury_goods", "electronics"])] = 3.0
    amount_mult[merchant == "gift_cards"] = 2.0
    amount = np.round(rng.lognormal(3.2, 0.9, n) * amount_mult, 2).clip(0.5, 50000)

    p_intl = np.where(merchant == "travel", 0.25, 0.03) + 0.25 * is_burst
    is_international = rng.binomial(1, np.clip(p_intl, 0, 1)).astype(int)
    p_dev = np.where(channel == "online", 0.07, 0.02) + 0.35 * is_burst
    device_new = rng.binomial(1, np.clip(p_dev, 0, 1)).astype(int)
    account_age_days = (card_age0[card_idx] + seconds // 86400).astype(float)

    high_risk = np.isin(merchant, HIGH_RISK_CATEGORIES).astype(float)
    is_night = (hour < 6).astype(float)
    log_amt = np.log(amount)
    logit = (INTERCEPT
             + 1.4 * high_risk
             + 1.8 * is_international
             + 1.8 * device_new
             + 0.5 * (channel == "online")
             + 0.7 * is_night
             + 1.0 * np.clip(log_amt - 3.5, 0, None)
             + 1.0 * (account_age_days < 90)
             + 2.0 * is_burst)
    p = 1 / (1 + np.exp(-logit))
    is_fraud = rng.binomial(1, p)

    df = pd.DataFrame({
        "transaction_id": [f"T{i:08d}" for i in range(n)],
        "timestamp": timestamp,
        "customer_id": [f"C{c:05d}" for c in card_idx],
        "amount": amount,
        "merchant_category": merchant,
        "channel": channel,
        "is_international": is_international,
        "device_new": device_new,
        "account_age_days": account_age_days,
        "is_fraud": is_fraud,
    })
    # Realistic data-quality issues
    df.loc[rng.random(n) < 0.015, "account_age_days"] = np.nan  # ~1.5% missing
    df.loc[rng.random(n) < 0.005, "merchant_category"] = np.nan
    return df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(config.RAW_PATH))
    ap.add_argument("--cards", type=int, default=5000)
    args = ap.parse_args()
    df = generate(n_cards=args.cards)
    config.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} transactions to {args.out}")
    print(f"Fraud rate: {df[config.TARGET].mean():.4%}  ({df[config.TARGET].sum():,} frauds)")


if __name__ == "__main__":
    main()
