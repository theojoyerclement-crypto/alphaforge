# AlphaForge

**Framework de recherche alpha quantitative pour les marchés crypto.**
Ingestion des données → features causales → découverte d'alphas sous contrôle des tests
multiples → familles de stratégies → construction de portefeuille sous contrainte de risque →
backtest événementiel avec frictions réalistes → ML en walk-forward → reporting. Le tout est
piloté par un fichier de configuration typé et reproductible au commit, à la seed et au hash
des données près.

🇬🇧 [English version](README.md) · 📐 [Architecture](docs/ARCHITECTURE.md) · 🔬 [Méthodologie](docs/METHODOLOGY.md)

---

## Pourquoi ce projet

Un backtester classique répond à *« combien cette stratégie aurait-elle gagné ? »*. AlphaForge
répond à la question que pose réellement un desk de recherche : **« cet edge est-il réel, et
survit-il à l'exécution ? »**

| Biais classique | Protection dans AlphaForge |
|---|---|
| Look-ahead bias | Chaque feature et chaque stratégie à règles est **testée par propriété** : on brouille le futur, le passé ne doit pas bouger. |
| Exécution sur la même barre | Un signal à la clôture *t* est exécuté à la clôture *t + délai* (`execution_delay: 1` par défaut). |
| Exécution sans friction | Frais + demi-spread + **impact de marché en racine carrée** (volatilité × participation à l'ADV) + **funding des perpétuels** barre par barre. |
| Turnover mal mesuré | Le turnover est mesuré par rapport aux poids **après dérive des prix**, ce qu'il faut réellement traiter. |
| ML évalué in-sample | Toute stratégie qui apprend est **refusée** hors walk-forward ; chaque fold réentraîne des modèles neufs. |
| Fuite des labels en CV | **Purged K-fold + embargo** et **CV combinatoire purgée** (López de Prado). |
| Data snooping sur des dizaines d'alphas | t-stats **Newey-West**, contrôle du FDR **Benjamini-Hochberg**, **Deflated Sharpe Ratio**. |
| Résultats non reproductibles | Chaque run enregistre config résolue, hash de config, hash des données, commit git, seed, versions des packages. |

## Résultats sur les configurations de référence

Le dépôt inclut un marché synthétique de perpétuels déterministe (queues épaisses, volatilité
GARCH, régimes, cotations décalées) qui **implante trois edges faibles et connus** : tendance,
retournement à 1 barre et carry de funding. C'est aussi un test d'intégration : une chaîne de
recherche qui ne retrouve pas les edges implantés, ou qui en « découvre » d'absents, est cassée.

**1. La recherche d'alphas retrouve ce qui a été implanté et rejette le reste**

| Alpha | IC (1 barre) | t Newey-West | BH à 10 % | Deflated Sharpe | Verdict |
|---|---:|---:|:---:|---:|---|
| `reversal_1` | 0,039 | 5,5 | ✅ | 0,99 | implanté ✔ |
| `basis_carry_5` | 0,022 | 3,1 | ✅ | 0,72 | implanté ✔ |
| `funding_carry_21` | 0,018 | 2,7 | ✅ | 0,68 | implanté ✔ |
| `mom_60` | 0,019 | 2,7 | ✅ | 0,26 | implanté, faible ✔ |
| `low_vol_60` | 0,007 | 0,9 | ❌ | 0,00 | non implanté ✔ |

**2. Un alpha significatif n'est pas une stratégie exploitable** (même alpha de retournement)

| Hypothèse d'exécution | Sharpe | CAGR |
|---|---:|---:|
| Sans friction, même barre | **+1,83** | +39,4 % |
| + frais, spread, impact | −0,28 | −6,8 % |
| + 1 barre de latence (défaut) | **−2,91** | −45,5 % |

**3. Portefeuille multi-stratégies** (net de tous les coûts, juil. 2019 – juin 2024)

| | CAGR | Vol | Sharpe | Sortino | Max DD |
|---|---:|---:|---:|---:|---:|
| **Portefeuille (risk parity, cible 20 % de vol)** | **9,5 %** | 15,8 % | **0,65** | 0,95 | −27,2 % |
| Carry de funding | 36,3 % | 16,0 % | 2,02 | 3,10 | −15,2 % |
| Momentum cross-sectionnel | 5,8 % | 15,1 % | 0,45 | 0,63 | −24,7 % |
| Momentum time-series | 0,7 % | 15,3 % | 0,12 | 0,17 | −20,2 % |

**4. ML en walk-forward** (11 folds hors échantillon) : le modèle directionnel LightGBM
(Sharpe −0,34) et le meta-labeling (≈ 0) **ne battent pas** une simple règle de carry (0,70) ;
la carte régime → stratégie apprise par GMM atteint 0,31. Ces résultats sont publiés tels quels,
sans réglage a posteriori. C'est précisément ce que le framework doit montrer.

## Démarrage rapide

```bash
python -m venv .venv && .venv\Scripts\activate        # Linux/macOS : source .venv/bin/activate
pip install -e ".[all]"

pytest                                                # 141 tests
alphaforge list                                       # stratégies, features, alphas
alphaforge research configs/multi_strategy.yaml       # classement des alphas
alphaforge run configs/multi_strategy.yaml            # backtest + rapport HTML dans runs/
alphaforge run configs/ml_walk_forward.yaml           # portefeuille ML en walk-forward
alphaforge run configs/multi_strategy.yaml --set portfolio.target_vol=0.15
```

Données réelles (perpétuels USDT-M Binance, avec funding) :

```bash
alphaforge fetch --exchange binance --symbols BTC/USDT:USDT ETH/USDT:USDT --timeframe 1d --start 2021-01-01
alphaforge run configs/binance_perps.yaml
```

## Architecture

| Module | Rôle |
|---|---|
| `data` | Ingestion CCXT (OHLCV + funding, pagination, retries), générateur synthétique, nettoyage audité, stockage Parquet avec métadonnées hashées, panels `MarketData` alignés |
| `features` | 21 features causales : momentum, estimateurs de volatilité, RSI, efficiency ratio ; proxies de microstructure (Amihud, Roll, Corwin-Schultz, λ de Kyle) ; funding et basis |
| `alpha` | Bibliothèque de 15 alphas, IC de rang et décroissance, spreads par quantile, Newey-West, PSR/DSR, Benjamini-Hochberg, combinaison causale pondérée par l'IC |
| `strategies` | Momentum TS/XS, mean reversion, retournement court terme, régimes, carry, composite d'alphas, ML directionnel, meta-labeling, régimes GMM |
| `portfolio` | Risque EWMA, sizing inverse-vol, ciblage de volatilité ex-ante, risk parity ERC, pénalité de corrélation, limites, contrôle du drawdown, Kelly fractionnaire |
| `backtest` | Moteur événementiel, modèles de coûts linéaire et racine carrée, walk-forward, purged K-fold et CPCV |
| `ml` | Labels à horizon fixe et triple barrière, modèle directionnel poolé, meta-labeling avec bet sizing, régimes GMM et de volatilité |
| `analytics` | Sharpe, Sortino, Calmar, profit factor, VaR/CVaR, épisodes de drawdown, table mensuelle, graphiques, rapport HTML autonome |
| `research` | `run_experiment` / `run_alpha_research` : config → dossier de résultats avec manifeste |

Détails de conception : [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Méthodologie et contrôles
de biais : [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Avertissement

Logiciel de recherche. Rien ici ne constitue un conseil en investissement. Les résultats sur
données synthétiques illustrent la méthodologie, pas une performance réelle.
