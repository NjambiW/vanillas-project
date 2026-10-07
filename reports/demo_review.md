# Demo forward test review

Generated 2026-10-07T11:34:43Z  
Journal: `C:\Users\benne\PycharmProjects\vanillas project\logs\journal.db`

## Overall (36/100 settled trades)

- Win rate: 27.8% (95% interval 13.1% to 42.4%)
- Total P&L: -278.38
- Expectancy per trade (money): -7.7328
- Avg win / avg loss: 32.921 / -23.369
- Profit factor: 0.54
- Max drawdown (money): 278.38
- Longest losing streak: 8
- Return per 1.0 staked (after markup): -30.9% +/- 21.4%
- Drawdown in stake units: 11.14
- Return before markup: -23.4% (t = -0.99; within +/-2 means no sign of an edge)
- Unsettled / unknown: 1
- Signals logged: 94

## Backtest reference

- none supplied

## Breakdowns

**strategy**
- trend_pullback: 36 trades | win 27.8% | exp -7.7328 | PF 0.54

**side**
- VANILLALONGPUT: 21 trades | win 19.0% | exp -15.1057 | PF 0.19
- VANILLALONGCALL: 15 trades | win 40.0% | exp +2.5893 | PF 1.18

**symbol**
- 1HZ100V: 36 trades | win 27.8% | exp -7.7328 | PF 0.54

**duration**
- 5m: 36 trades | win 27.8% | exp -7.7328 | PF 0.54

**hour_utc**
- 01:00: 1 trades | win 0.0% | exp -25.0000 | PF 0.00
- 02:00: 3 trades | win 66.7% | exp +1.1567 | PF 1.14
- 03:00: 2 trades | win 50.0% | exp +51.8400 | PF 5.15
- 04:00: 3 trades | win 33.3% | exp -9.7767 | PF 0.34
- 05:00: 3 trades | win 0.0% | exp -25.0000 | PF 0.00
- 06:00: 2 trades | win 0.0% | exp -25.0000 | PF 0.00
- 07:00: 1 trades | win 0.0% | exp -25.0000 | PF 0.00
- 08:00: 1 trades | win 0.0% | exp -25.0000 | PF 0.00
- 09:00: 1 trades | win 100.0% | exp +17.0300 | PF n/a
- 10:00: 1 trades | win 100.0% | exp +6.7200 | PF n/a
- 11:00: 3 trades | win 0.0% | exp -25.0000 | PF 0.00
- 14:00: 1 trades | win 0.0% | exp -25.0000 | PF 0.00
- 15:00: 1 trades | win 100.0% | exp +3.6300 | PF n/a
- 18:00: 2 trades | win 0.0% | exp -25.0000 | PF 0.00
- 20:00: 5 trades | win 20.0% | exp -6.7680 | PF 0.57
- 21:00: 3 trades | win 66.7% | exp +23.2300 | PF 5.50
- 22:00: 1 trades | win 0.0% | exp -25.0000 | PF 0.00
- 23:00: 2 trades | win 0.0% | exp -22.2150 | PF 0.00

## Skipped signals (57)

- DRY RUN: would buy VANILLALONGPUT #m barrier +# stake # (markup #%): 24
- DRY RUN: would buy VANILLALONGCALL #m barrier +# stake # (markup #%): 17
- markup #% is above the #% limit: 12
- a trade is already open: 4

## Weakness list (candidates for our joint review)

- [INFO] Only 36/100 demo trades logged; sample too small for conclusions.
- [MED] 12 signals blocked by the premium gate; average quoted markup 17.3% (lowest 17.2%).
- [MED] 1 trades have no recorded profit (open or settlement unknown).
- [MED] strategy 'trend_pullback': 36 trades, expectancy -7.7328.
- [MED] side 'VANILLALONGPUT': 21 trades, expectancy -15.1057.
- [INFO] No backtest numbers supplied; comparison skipped. Create them with: python src/run_backtest.py --strategy NAME --hold N --export

Rule: change one thing at a time, then re-run the demo and this review.
