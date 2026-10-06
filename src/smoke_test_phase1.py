"""Phase 1 smoke test: connect, read balance, read one tick, price vanillas.

It does NOT buy anything. Run:  python src/smoke_test_phase1.py
"""
import asyncio
import json
import logging

from config import DURATION, DURATION_UNIT, STAKE, SYMBOL
from deriv_client import DerivAPIError, DerivClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


async def main() -> None:
    client = DerivClient()
    await client.start()
    try:
        bal = await client.get_balance()
        currency = bal["currency"]
        print(f"Balance: {bal['balance']} {currency}")

        first = await client.subscribe_ticks(SYMBOL, lambda msg: None)
        print(f"{SYMBOL} spot: {first['tick']['quote']}")
        sub_id = (first.get("subscription") or {}).get("id")
        if sub_id:
            await client.forget(sub_id)

        for contract_type in ("VANILLALONGCALL", "VANILLALONGPUT"):
            barriers = await client.get_allowed_barriers(
                contract_type, SYMBOL, DURATION, DURATION_UNIT, STAKE, currency
            )
            print(f"\n=== {contract_type}  {DURATION}{DURATION_UNIT}  stake {STAKE}")
            print(f"Allowed barriers: {barriers}")
            for i, barrier in enumerate(barriers):
                try:
                    proposal = await client.get_proposal(
                        contract_type, SYMBOL, barrier, DURATION, DURATION_UNIT, STAKE, currency
                    )
                except DerivAPIError as exc:
                    print(f"  {barrier}: API said no: {exc}")
                    continue
                if i == 0:
                    print("Full first proposal:")
                    print(json.dumps(proposal, indent=2))
                else:
                    short = {k: v for k, v in proposal.items() if k != "longcode"}
                    print(f"  {barrier}: {json.dumps(short)}")
                await asyncio.sleep(0.2)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())