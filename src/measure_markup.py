"""Phase 3: measure how much Deriv charges above the fair Black-Scholes price.

For each duration and each allowed strike it requests a vanilla proposal, then
compares the price with the model. Nothing is bought. Results go to logs/.

Run:  python src/measure_markup.py
"""
import asyncio
import csv
import json
import logging
from datetime import datetime

from config import LOG_DIR, STAKE, SYMBOL
from deriv_client import DerivAPIError, DerivClient
from markup import MarkupError, markup_row

logging.basicConfig(level=logging.WARNING)

DURATIONS = [(1, "m"), (5, "m"), (15, "m"), (1, "h"), (4, "h"), (1, "d")]
CONTRACT_TYPES = ("VANILLALONGCALL", "VANILLALONGPUT")


async def main() -> None:
    client = DerivClient()
    await client.start()
    rows: list[dict] = []
    first_raw_saved = False
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        currency = (await client.get_balance())["currency"]
        first = await client.subscribe_ticks(SYMBOL, lambda m: None)
        spot = float(first["tick"]["quote"])
        sub_id = (first.get("subscription") or {}).get("id")
        if sub_id:
            await client.forget(sub_id)
        print(f"{SYMBOL} spot {spot}, stake {STAKE} {currency}\n")

        for duration, unit in DURATIONS:
            for contract_type in CONTRACT_TYPES:
                try:
                    barriers = await client.get_allowed_barriers(
                        contract_type, SYMBOL, duration, unit, STAKE, currency
                    )
                except DerivAPIError as exc:
                    print(f"{contract_type} {duration}{unit}: not available ({exc})")
                    continue
                for barrier in barriers:
                    try:
                        proposal = await client.get_proposal(
                            contract_type, SYMBOL, barrier, duration, unit, STAKE, currency
                        )
                    except DerivAPIError as exc:
                        print(f"{contract_type} {duration}{unit} {barrier}: {exc}")
                        continue
                    if not first_raw_saved:
                        path = LOG_DIR / f"raw_proposal_{stamp}.json"
                        path.write_text(json.dumps(proposal, indent=2))
                        print(f"(raw proposal saved to {path})")
                        first_raw_saved = True
                    try:
                        rows.append(
                            markup_row(contract_type, proposal, spot, barrier, duration, unit, SYMBOL)
                        )
                    except MarkupError as exc:
                        print(f"STOP: {exc}")
                        print(json.dumps(proposal, indent=2))
                        return
                    await asyncio.sleep(0.25)
    finally:
        await client.close()

    if not rows:
        print("No proposals were priced.")
        return

    header = f"{'type':16}{'dur':>5}{'barrier':>9}{'ask':>8}{'fair':>9}{'markup%':>9}{'impl.vol':>10}"
    print("\n" + header)
    for r in rows:
        print(
            f"{r['type']:16}{r['duration']:>5}{r['barrier']:>9}{r['ask_price']:>8.2f}"
            f"{r['fair_price']:>9.4f}{r['markup_pct']:>9.1f}{r['implied_vol']:>10.3f}"
        )

    usable = [r for r in rows if r["markup_pct"] == r["markup_pct"]]
    usable.sort(key=lambda r: r["markup_pct"])
    print("\nLowest markup:")
    for r in usable[:5]:
        print(f"  {r['type']} {r['duration']} barrier {r['barrier']}: {r['markup_pct']}%")

    out = LOG_DIR / f"markup_{stamp}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out}")


if __name__ == "__main__":
    asyncio.run(main())