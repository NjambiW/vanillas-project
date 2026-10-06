"""Buy ONE small contract on your DEMO account to prove buying and settlement work.

This tests plumbing, not profitability. It buys a 1-minute at-the-money Call at a tiny stake,
prints every message Deriv sends, and saves them to logs/ so the exact field names can be checked.

Run:  python src/demo_trade_check.py            (asks you to confirm first)
"""
import argparse
import asyncio
import json
from datetime import datetime

from config import CLIENTid, LOG_DIR, SYMBOL
from deriv_client import DerivClient
from executor import _truthy, pick_barrier, seconds_to_duration
from markup import MarkupError, markup_row


async def check(client, symbol, stake, seconds, confirm, out_dir=None, settle_buffer=90):
    record = {"symbol": symbol, "stake": stake}
    duration, unit = seconds_to_duration(seconds)

    bal = await client.get_balance()
    currency = bal["currency"]
    print(f"Account {CLIENTid}  balance {bal['balance']} {currency}")
    if not confirm(CLIENTid, bal["balance"]):
        print("Cancelled. Nothing was bought.")
        return {"cancelled": True}

    first = await client.subscribe_ticks(symbol, lambda m: None)
    spot = float(first["tick"]["quote"])
    sub_id = (first.get("subscription") or {}).get("id")
    if sub_id:
        await client.forget(sub_id)

    allowed = await client.get_allowed_barriers("VANILLALONGCALL", symbol, duration, unit, stake, currency)
    barrier = pick_barrier(allowed, 0.0, spot)
    proposal = await client.get_proposal("VANILLALONGCALL", symbol, barrier, duration, unit, stake, currency)
    record["proposal"] = proposal
    try:
        row = markup_row("VANILLALONGCALL", proposal, spot, barrier, duration, unit, symbol)
        print(f"Priced: ask {row['ask_price']}  strike {row['strike']}  markup {row['markup_pct']}%")
    except MarkupError as exc:
        print(f"(could not work out the markup: {exc})")

    bought = await client.buy(proposal["id"], float(proposal["ask_price"]))
    record["buy"] = bought
    print("\nBUY REPLY (raw):")
    print(json.dumps(bought, indent=2, default=str))
    contract_id = bought.get("contract_id")

    loop = asyncio.get_running_loop()
    done = loop.create_future()
    updates = []

    def on_update(msg):
        poc = msg.get("proposal_open_contract") or {}
        updates.append(poc)
        print(f"  update: is_sold={poc.get('is_sold')} is_expired={poc.get('is_expired')} "
              f"status={poc.get('status')} profit={poc.get('profit')} spot={poc.get('current_spot')}")
        if (_truthy(poc.get("is_sold")) or _truthy(poc.get("is_expired"))) and not done.done():
            done.set_result(poc)

    first_poc = await client.subscribe({"proposal_open_contract": 1, "contract_id": contract_id}, on_update)
    record["first_open_contract_message"] = first_poc
    sub = (first_poc.get("subscription") or {}).get("id")
    try:
        final = await asyncio.wait_for(done, seconds + settle_buffer)
    except asyncio.TimeoutError:
        print("Timed out waiting for settlement. Check the contract in your Deriv account.")
        final = None
    finally:
        if sub:
            await client.forget(sub)

    record["final_open_contract"] = final
    record["updates_seen"] = len(updates)
    if final is not None:
        print(f"\nSETTLED. profit = {final.get('profit')}   (final message saved to the log file)")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"demo_trade_check_{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps(record, indent=2, default=str))
        print(f"Raw messages saved to {path}")
    return record


def ask_confirmation(account, balance) -> bool:
    print("\nThis will BUY one real contract through the API on the account above.")
    print("Only continue if that is your DEMO account.")
    return input("Type YES to continue: ").strip() == "YES"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stake", type=float, default=0.5)
    ap.add_argument("--seconds", type=int, default=60)
    args = ap.parse_args()
    client = DerivClient()
    await client.start()
    try:
        await check(client, SYMBOL, args.stake, args.seconds, ask_confirmation, LOG_DIR)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())