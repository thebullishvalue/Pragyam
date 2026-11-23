<div align="center">

# PRAGYAM

**प्रज्ञम्**

Portfolio Curation · Equal Weight · ERC · HRP · Live NSE Data

[![Version](https://img.shields.io/badge/version-13.0.22-2B5FD9)](CHANGELOG.md) [![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](requirements.txt) [![Streamlit](https://img.shields.io/badge/streamlit-1.52.2-FF4B4B)](requirements.txt) [![License](https://img.shields.io/badge/license-proprietary-6B7280)](LICENSE) [![Pragyam Family](https://img.shields.io/badge/pragyam-family-0F1217)](#the-pragyam-family)

</div>

> The system curates a book by measuring how its holdings move together, then shows what each
> one contributes to risk against what it costs in capital.

## Overview

Portfolio curation over a chosen universe — the ETF book by default, or an index,
commodity, crypto or custom list, in three styles: Equal Weight, Equal Risk
Contribution and Hierarchical Risk Parity. None of them forecasts a return; they
are built to **spread risk**, not to predict returns.

## What It Does

**Styles**

| Style | Family | Behaviour |
|---|---|---|
| **Equal Weight** *(default)* | baseline | Identical `1/N` per holding. The default because nothing beat it reproducibly; see below. Lowest turnover of any style. |
| **Equal Risk Contribution** | preservation | Solves so every holding contributes the same share of portfolio variance. The preferred risk-reduction style on return: beats HRP on the any-date hit rate in 6 of 6 cells across two stock universes, at about 0.6x its turnover. In-repo (every name held, net, 2007-26): −0.49 %/yr against Equal Weight on Nifty 50 and −1.27 on Dow 30, at volatility 20.1 / 15.3 against 22.2 / 16.5. |
| **Risk Parity (HRP)** | preservation | Clusters by correlation distance, then splits capital by recursive bisection on cluster variance. Inverts no matrix. Averaged over three staggered estimation windows (v12.2). The deepest volatility and drawdown cut of the styles (volatility 18.8 / 14.2, max drawdown −49.3% / −36.4% on Nifty 50 / Dow 30), at −0.54 / −2.47 %/yr against Equal Weight and about 1.7× ERC's turnover. Measured holding every name: at 30 of 50 positions it is the 30 lowest-variance names (19.11 %/yr against 19.56). |

Every style travels the identical pipeline — same eligibility filter, same
clustering diagnostics, same risk decomposition — so any difference on screen is
the allocator and nothing else.

**Result tabs**

| Tab | Contents |
|---|---|
| **Portfolio** | Holdings with weight, risk share, volatility, independence · risk-profile heatmap · **risk-contribution chart** · cluster correlation matrix |
| **Analytics** | Head-to-head table (book / EW Shadow / benchmark), a **Style Comparison** of the book each of the other two styles builds from the same run, and benchmark-relationship statistics, over an indexed performance chart |
| **Regime** | 8-factor composite + history. Context only — nothing is conditioned on it |
| **Broker Sync** | Writes curated units into broker order-template JSONs |
| **System** | Configuration, methodology, execution metrics |

## Quick Start

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

**Sidebar:** Analysis Date · Portfolio Style · Universe · Capital · Positions.

## Project Structure

```
app.py          Streamlit UI + 2-phase pipeline
nco.py          Equal Weight / ERC / HRP curation
regime.py       8-factor regime detection (context only)
backdata.py     yfinance fetch + indicator panel
analytics.py    portfolio-vs-benchmark metrics
charts.py       Plotly builders
universe.py     universe resolution
research/       offline evidence harness (not imported by the app)
```

**Pipeline:** Phase 1 data + regime → Phase 2 curation. About 2s once the panel
is cached.

## Documentation

| Document | What it covers |
|---|---|
| [Guide](docs/GUIDE.md) | Why Equal Weight is the default, reading the Portfolio and Analytics tabs, known limits |
| [Release notes](docs/RELEASE_NOTES.md) | What changed in v11 through v13, and why |
| [Research](research/) | The offline evidence harness behind the ERC and HRP figures (not imported by the app) |
| [Changelog](CHANGELOG.md) | Every release, newest first |

## The Pragyam Family

Ten Streamlit applications built on one design system and one set of components, so a
panel, a table, a chart or a number reads the same in each.

| Product | What it is |
|---|---|
| [**Tattva**](https://github.com/thebullishvalue/Tattva) | Cross-asset fair value, with a call only where its own record supports one |
| [**Nishkarsh**](https://github.com/thebullishvalue/Nishkarsh) | Nifty 50 inference: a top-down forecast and bottom-up breadth, read as one |
| [**Arthagati**](https://github.com/thebullishvalue/Arthagati) | Valuation-anchored market sentiment, tested on data it never saw |
| [**Vyapti**](https://github.com/thebullishvalue/Vyapti) | Market breadth: where participation stands, which way it moves, and what that has meant |
| **Pragyam** · *this repository* | Portfolio curation that spreads risk rather than predicting returns |
| [**Sanket**](https://github.com/thebullishvalue/Sanket) | A market signal screener: what fired, what is about to, and whether it has paid |
| [**Vaaydo**](https://github.com/thebullishvalue/Vaaydo) | An F&O trade screener on option positioning, with futures tickets |
| [**Nivesa**](https://github.com/thebullishvalue/Nivesa) | A bond portfolio ledger: cost, yield, duration and cashflows across accounts |
| [**Samhita**](https://github.com/thebullishvalue/Samhita) | A portfolio tracker: live prices, performance and risk against the benchmark |
| [**Swing**](https://github.com/thebullishvalue/Swing) | A portfolio tracker for the swing book, on the same engine as Samhita |

## License

Proprietary. © 2024–2026 @thebullishvalue. All rights reserved. The source is visible here
for reference only: viewing it grants no right to use, copy, modify or distribute it.
See [LICENSE](LICENSE).

## Disclaimer

Pragyam is a research and decision-support tool, not investment advice. Nothing it shows is
a recommendation to buy or sell any security, and past or measured results do not guarantee
future ones. Every decision made with it is your own. None of its styles forecasts a
return.

---

<div align="center"><sub>© 2024–2026 @thebullishvalue · Part of the Pragyam product family</sub></div>
