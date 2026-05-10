#!/usr/bin/env python3
# coding: utf-8
"""
Gemma 4 31B via Ollama (GGUF Q4_K_M) + asyncio
Setup: Ollama running with OLLAMA_NUM_PARALLEL=4
"""

import asyncio
import base64
import json
import os
import re
import time

import ollama
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_MODEL = "gemma4:31b"

FOLDER     = r"z:/Nutstore/GPT/SVXY"
PATTERN    = re.compile(r"(?P<date>\d{8})_(?P<type>daily|weekly)\.png", re.IGNORECASE)
START_DATE = 20240101

CONCURRENCY = 4

JSON_PATH = "forecast_predictions_with_prob_ollama_gemma4.json"
CSV_PATH  = r"c:/jiaqifiles/rdata/forecast_predictions_with_prob_ollama_gemma4.csv"

SYSTEM_MSG = "You are a financial analyst specializing in technical analysis."
PROMPT = (
    "Based on the OHLC candlestick chart, volume, and MACD, predict tomorrow's price direction.\n"
    "In the candlestick chart, green candles mean close > open (bullish), the other color means close < open (bearish).\n"
    "The first image is the daily chart, the second is the weekly chart.\n"
    "Requirements:\n"
    "1. Use your expertise in technical analysis to make a judgment. No explanation needed.\n"
    "2. Reply in exactly one of these three formats:\n"
    "   - Tomorrow may rise, probability 0.x (value between 0.1 and 1.0, one decimal place)\n"
    "   - Tomorrow may fall, probability -0.x (value between -0.1 and -1.0, one decimal place)\n"
    "   - Tomorrow is uncertain, probability 0\n"
    "3. Only one of the three formats above.\n"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_date_images(folder: str) -> dict:
    date_images: dict = {}
    for fname in os.listdir(folder):
        m = PATTERN.search(fname)
        if not m:
            continue
        date = m.group("date")
        chart_type = m.group("type").lower()
        date_images.setdefault(date, {})[chart_type] = os.path.join(folder, fname)
    return date_images


def encode_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def parse_prediction(raw_text: str) -> dict:
    m = re.search(r"(?:probability)\s*([-+]?[0-9]*\.?[0-9]+)", raw_text, re.IGNORECASE)
    pred_y = float(m.group(1)) if m else 0.0

    text_lower = raw_text.lower()
    if "rise" in text_lower or "up" in text_lower:
        direction = "up"
    elif "fall" in text_lower or "down" in text_lower:
        direction = "down"
    elif "uncertain" in text_lower:
        direction = "uncertain"
    else:
        direction = "unknown"

    return {"direction": direction, "pred_y": pred_y}


# ---------------------------------------------------------------------------
# Async inference
# ---------------------------------------------------------------------------

async def predict_one(
    client: ollama.AsyncClient,
    sem: asyncio.Semaphore,
    date: str,
    daily_path: str,
    weekly_path: str,
    counter: list,
    total: int,
) -> tuple[str, str]:
    async with sem:
        daily_b64  = encode_b64(daily_path)
        weekly_b64 = encode_b64(weekly_path)

        response = await client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {
                    "role": "user",
                    "content": PROMPT,
                    "images": [daily_b64, weekly_b64],
                },
            ],
            options={"num_predict": 50},
            think=False,
        )

        content = response.message.content or ""
        counter[0] += 1
        pct = counter[0] / total * 100
        print(f"[{counter[0]:4d}/{total}  {pct:5.1f}%]  {date}  ->  {content[:80].strip()}")
        return date, content


async def run_all(date_images: dict) -> dict:
    client = ollama.AsyncClient()
    sem    = asyncio.Semaphore(CONCURRENCY)

    jobs = [
        (date, imgs["daily"], imgs["weekly"])
        for date, imgs in sorted(date_images.items())
        if int(date) >= START_DATE
        and "daily" in imgs
        and "weekly" in imgs
    ]

    total   = len(jobs)
    counter = [0]
    print(f"Starting {total} predictions with concurrency={CONCURRENCY} ...\n")

    tasks = [
        predict_one(client, sem, date, dp, wp, counter, total)
        for date, dp, wp in jobs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    predictions: dict = {}
    for r in results:
        if isinstance(r, Exception):
            print(f"ERROR: {r}")
        else:
            date, content = r
            predictions[date] = content

    return predictions


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(predictions: dict) -> None:
    structured: dict = {}
    for date, raw in predictions.items():
        parsed = parse_prediction(raw)
        structured[date] = {"raw": raw, "direction": parsed["direction"], "pred_y": parsed["pred_y"]}

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON -> {JSON_PATH}")

    df = pd.DataFrame([
        {"date": d, "direction": v["direction"], "pred_y": v["pred_y"], "raw_text": v["raw"]}
        for d, v in structured.items()
    ])
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved CSV  -> {CSV_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.perf_counter()

    date_images = build_date_images(FOLDER)
    predictions = asyncio.run(run_all(date_images))

    elapsed = time.perf_counter() - t0
    n       = len(predictions)
    per_req = elapsed / n if n else 0

    print("\n" + "=" * 60)
    print(f"  Total predictions : {n}")
    print(f"  Total time        : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Avg per request   : {per_req:.2f}s")
    print(f"  Effective RPS     : {n/elapsed:.2f}" if elapsed > 0 else "")
    print("=" * 60)

    save_outputs(predictions)


if __name__ == "__main__":
    main()
