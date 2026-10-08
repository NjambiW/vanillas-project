# Demo forward test review

Generated 2026-10-07T19:40:43Z  
Journal: `C:\Users\benne\PycharmProjects\vanillas project\logs\journal.db`

## Overall (54/100 settled trades)

- Win rate: 25.9% (95% interval 14.2% to 37.6%)
- Total P&L: -509.20
- Expectancy per trade (money): -9.4296
- Avg win / avg loss: 30.515 / -23.410
- Profit factor: 0.46
- Max drawdown (money): 521.22
- Longest losing streak: 8
- Return per 1.0 staked (after markup): -37.7% +/- 16.1%
- Drawdown in stake units: 20.85
- Return before markup: -30.9% (t = -1.74; within +/-2 means no sign of an edge)
- Unsettled / unknown: 1
- Signals logged: 135

## Backtest reference

- roi: 0.023408235498212897
- trades: 123.0
- p_value: 0.2413793103448276
- hold_candles: 60.0
- variants_tested: 9.0
- expectancy: 0.023408235498212897
- strategy: mean_reversion
- cost_model_source: C:\Users\benne\PycharmProjects\vanillas project\logs\markup_20261006_113251.csv (+ built-in table for missing expiries)
- generated: 2026-10-07T18:16:06Z
- train_roi: 0.09013607075718891
- train_p_value: 0.07796101949025487

## Breakdowns

**strategy**
- trend_pullback: 54 trades | win 25.9% | exp -9.4296 | PF 0.46

**side**
- VANILLALONGPUT: 24 trades | win 20.8% | exp -14.7500 | PF 0.20
- VANILLALONGCALL: 30 trades | win 30.0% | exp -5.1733 | PF 0.68

**symbol**
- 1HZ100V: 54 trades | win 25.9% | exp -9.4296 | PF 0.46

**duration**
- 5m: 54 trades | win 25.9% | exp -9.4296 | PF 0.46

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
- 11:00: 5 trades | win 0.0% | exp -25.0000 | PF 0.00
- 12:00: 2 trades | win 0.0% | exp -25.0000 | PF 0.00
- 13:00: 3 trades | win 33.3% | exp +12.2833 | PF 2.28
- 14:00: 3 trades | win 33.3% | exp -12.2600 | PF 0.26
- 15:00: 4 trades | win 50.0% | exp -9.8200 | PF 0.21
- 16:00: 1 trades | win 0.0% | exp -25.0000 | PF 0.00
- 17:00: 4 trades | win 0.0% | exp -25.0000 | PF 0.00
- 18:00: 2 trades | win 0.0% | exp -25.0000 | PF 0.00
- 19:00: 1 trades | win 100.0% | exp +12.0200 | PF n/a
- 20:00: 5 trades | win 20.0% | exp -6.7680 | PF 0.57
- 21:00: 3 trades | win 66.7% | exp +23.2300 | PF 5.50
- 22:00: 1 trades | win 0.0% | exp -25.0000 | PF 0.00
- 23:00: 2 trades | win 0.0% | exp -22.2150 | PF 0.00

## Skipped signals (80)

- DRY RUN: would buy VANILLALONGCALL #m barrier +# stake # (markup #%): 33
- DRY RUN: would buy VANILLALONGPUT #m barrier +# stake # (markup #%): 28
- markup #% is above the #% limit: 12
- a trade is already open: 5
- # losses in a row; stopped until the next UTC day: 2

## Weakness list (candidates for our joint review)

- [INFO] Only 54/100 demo trades logged; sample too small for conclusions.
- [MED] 12 signals blocked by the premium gate; average quoted markup 17.3% (lowest 17.2%).
- [HIGH] Demo ran ['trend_pullback'] but the backtest report is for 'mean_reversion'. They cannot be compared.
- [HIGH] Return per stake changes sign: demo -37.7% vs backtest +2.3%.
- [MED] 1 trades have no recorded profit (open or settlement unknown).
- [MED] strategy 'trend_pullback': 54 trades, expectancy -9.4296.
- [MED] side 'VANILLALONGPUT': 24 trades, expectancy -14.7500.
- [MED] side 'VANILLALONGCALL': 30 trades, expectancy -5.1733.

Rule: change one thing at a time, then re-run the demo and this review.
