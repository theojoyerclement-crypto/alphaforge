# AlphaForge

**Framework de recherche alpha quantitative pour les marchés crypto.**
Ingestion des données → features causales → découverte d'alphas sous contrôle des tests
multiples → familles de stratégies → construction de portefeuille sous contrainte de risque →
backtest événementiel avec frictions réalistes → ML en walk-forward → paper trading sur le même
code → reporting. Le tout est piloté par un fichier de configuration typé et reproductible au
commit, à la seed et au hash des données près.

🇬🇧 [English version](README.md) · 📐 [Architecture](docs/ARCHITECTURE.md) · 🔬 [Méthodologie](docs/METHODOLOGY.md) · 🤝 [Contribuer](CONTRIBUTING.md)

![python](https://img.shields.io/badge/python-3.11%2B-1f4e79) ![tests](https://img.shields.io/badge/tests-161%20passing-4f8a3c) ![license](https://img.shields.io/badge/license-MIT-555555) ![status](https://img.shields.io/badge/status-research-8064a2)

---

## Ce qu'un recruteur devrait regarder (10 minutes)

| À regarder | Pourquoi c'est important | Où |
|---|---|---|
| **Une seule commande** | Tests, backtest de référence et replay paper-trading, hors ligne, sur tout OS | `python scripts/smoke_test.py` |
| **Tests anti look-ahead** | Le futur est brouillé et chaque feature, chaque stratégie et le constructeur de portefeuille doivent donner le même passé | `tests/test_features.py`, `tests/test_strategies.py`, `tests/test_live.py` |
| **Identités comptables** | `net = brut + funding − coûts`, vérifié contre des carnets calculés à la main | `tests/test_backtest.py` |
| **Résultats sur données réelles, y compris les mauvais** | Perpétuels Binance, nets de frais, spread, impact et funding, comparés au buy & hold | [Données réelles](#5-marché-réel--perpétuels-binance-usdt-m) |
| **Un alpha significatif qui perd de l'argent** | t = 5,5, Sharpe +1,8 sans friction, −2,9 avec coûts et latence | [Résultats §2](#résultats) |
| **Un ML qui ne gagne pas** | Le ML en walk-forward perd face à une règle de carry, et le README le dit | [Résultats §4](#résultats) |
| **Parité backtest ↔ live** | La boucle paper recalcule les décisions barre par barre et retrouve l'equity du backtest à 1e-10 | `alphaforge paper ... --replay N`, `src/alphaforge/live/` |
| **Configs** | Un fichier YAML par expérience. Une clé inconnue est rejetée et l'erreur liste les clés valides | `configs/`, `src/alphaforge/config/schema.py` |
| **Choix de conception** | Les décisions qui comptent, documentées | [ARCHITECTURE](docs/ARCHITECTURE.md#design-choices-that-matter-to-a-hiring-manager) · [METHODOLOGY](docs/METHODOLOGY.md#design-choices-that-matter-to-a-hiring-manager) |

## Pourquoi ce projet

Un backtester classique répond à *« combien cette stratégie aurait-elle gagné ? »*. AlphaForge
répond à la question que pose réellement un desk de recherche : **« cet edge est-il réel, et
survit-il à l'exécution ? »**

| Biais classique | Protection dans AlphaForge |
|---|---|
| Look-ahead bias | Chaque feature et chaque stratégie à règles est **testée par propriété** : on brouille le futur, le passé ne doit pas bouger. |
| Exécution sur la même barre | Un signal à la clôture *t* est exécuté à la clôture *t + délai* (`execution_delay: 1` par défaut). |
| Exécution sans friction | Frais + demi-spread + **impact de marché en racine carrée** (volatilité × participation à l'ADV) + **funding des perpétuels** barre par barre. |
| Turnover mal mesuré | Le turnover est mesuré par rapport aux poids **après dérive des prix**, c'est-à-dire ce qu'il faut réellement traiter. |
| Bougies non clôturées | La bougie en cours de formation est écartée à l'ingestion et n'est jamais mise en cache comme une clôture définitive. |
| ML évalué in-sample | Toute stratégie qui apprend est **refusée** hors walk-forward ; chaque fold réentraîne des modèles neufs. |
| Fuite des labels en CV | **Purged K-fold + embargo** et **CV combinatoire purgée** (López de Prado). |
| Data snooping sur des dizaines d'alphas | t-stats **Newey-West**, contrôle du FDR **Benjamini-Hochberg**, **Deflated Sharpe Ratio**. |
| Code de backtest ≠ code live | Le paper trader appelle les mêmes stratégies, le même constructeur de portefeuille, le même modèle de coûts et la même fonction de mark-to-market que le moteur. Un test vérifie que les deux donnent la même equity. |
| Résultats non reproductibles | Chaque run enregistre config résolue, hash de config, hash des données, commit git (avec un indicateur d'arbre modifié), seed et versions des packages. |

## Résultats

Deux types de preuves, volontairement séparés :

* **Marché synthétique (sections 1 à 4).** Un marché de perpétuels déterministe (queues
  épaisses, volatilité GARCH, régimes, cotations décalées) qui **implante trois edges faibles et
  connus** : tendance, retournement à 1 barre et carry de funding. C'est un test d'intégration
  dont on connaît la réponse : une chaîne de recherche qui ne retrouve pas les edges implantés,
  ou qui en « découvre » d'absents, est cassée.
* **Marché réel (section 5).** Perpétuels USDT-M Binance avec le funding réel. C'est là qu'on
  voit ce que valaient vraiment les edges implantés.

**1. La recherche d'alphas retrouve ce qui a été implanté et rejette le reste**
(`alphaforge research configs/multi_strategy.yaml`, 12 actifs × 2000 barres journalières, 15 candidats)

| Alpha | IC (1 barre) | t Newey-West | BH à 10 % | Deflated Sharpe | Verdict |
|---|---:|---:|:---:|---:|---|
| `reversal_1` | 0,039 | 5,5 | ✅ | 0,99 | implanté ✔ |
| `basis_carry_5` | 0,022 | 3,1 | ✅ | 0,72 | implanté ✔ |
| `funding_carry_21` | 0,018 | 2,7 | ✅ | 0,68 | implanté ✔ |
| `mom_60` | 0,019 | 2,7 | ✅ | 0,26 | implanté, faible ✔ |
| `low_vol_60` | 0,007 | 0,9 | ❌ | 0,00 | non implanté ✔ |

**2. Un alpha significatif n'est pas une stratégie exploitable** (`configs/cost_stress.yaml`, même alpha de retournement)

| Hypothèse d'exécution | Sharpe | CAGR |
|---|---:|---:|
| Sans friction, même barre | **+1,83** | +39,4 % |
| + frais, spread, impact | −0,28 | −6,8 % |
| + 1 barre de latence (défaut) | **−2,91** | −45,5 % |

À ~460× de turnover annuel, les coûts consomment l'edge, et le retournement a déjà eu lieu quand
l'ordre est exécuté.

**3. Portefeuille multi-stratégies** (`configs/multi_strategy.yaml`, net de tous les coûts, juil. 2019 – juin 2024)

| | CAGR | Vol | Sharpe | Sortino | Max DD | Turnover/an |
|---|---:|---:|---:|---:|---:|---:|
| **Portefeuille (risk parity, cible 20 % de vol)** | **9,5 %** | 15,8 % | **0,65** | 0,95 | −27,2 % | 119× |
| Carry de funding | 36,3 % | 16,0 % | 2,02 | 3,10 | −15,2 % | 51× |
| Momentum cross-sectionnel | 5,8 % | 15,1 % | 0,45 | 0,63 | −24,7 % | 76× |
| Momentum time-series | 0,7 % | 15,3 % | 0,12 | 0,17 | −20,2 % | 49× |
| Mean reversion (z-score) | −2,7 % | 16,4 % | −0,09 | −0,13 | −31,6 % | 102× |
| Changement de régime | −4,1 % | 16,1 % | −0,18 | −0,26 | −45,8 % | 144× |

**4. ML en walk-forward** (`configs/ml_walk_forward.yaml`, 11 folds hors échantillon, réentraînement toutes les 120 barres)

| Sleeve (hors échantillon uniquement) | Sharpe | Turnover/an | Remarque |
|---|---:|---:|---|
| Modèle directionnel (gradient boosting) | −0,34 | 108× | pas d'edge à 5 barres une fois les coûts payés |
| Trend avec meta-labeling | 0,01 | 35× | dans `examples/03`, divise le turnover par ~14 et le max drawdown par ~5 |
| Carte régime GMM → stratégie | 0,31 | 117× | apprend « reversion en régime calme, carry sinon » |
| Carry de funding (référence) | 0,70 | 54× | le simple bat le complexe ici |

Ces résultats sont publiés tels quels, sans réglage a posteriori : le ML **ne bat pas** une
simple règle de carry sur ces données, et le framework le dit.

### 5. Marché réel : perpétuels Binance USDT-M

`configs/binance_perps.yaml` **n'a pas été modifié depuis le commit initial** et n'a donc pas été
réglé sur ces données. Univers : 12 grands perpétuels (BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX,
LINK, DOT, LTC, ATOM), barres journalières, funding réel toutes les 8 h, évaluation de juil. 2021
à oct. 2026 (données jusqu'au 2026-10-12), frictions par défaut (frais 4 bps, demi-spread
1,5 bps, impact racine carrée, 1 barre de latence).

| Net de tous coûts et du funding | CAGR | Vol | Sharpe | Sortino | Max DD | Turnover/an |
|---|---:|---:|---:|---:|---:|---:|
| **Portefeuille (risk parity, cible 20 % de vol)** | **6,1 %** | 18,0 % | **0,42** | 0,60 | −27,4 % | 121× |
| Momentum time-series | 5,5 % | 15,5 % | 0,43 | 0,61 | −22,9 % | 38× |
| Carry de funding | 5,1 % | 16,7 % | 0,38 | 0,53 | −28,8 % | 60× |
| Momentum cross-sectionnel | 2,1 % | 17,0 % | 0,21 | 0,29 | −30,7 % | 91× |
| Changement de régime | −11,5 % | 16,1 % | −0,68 | −0,93 | −49,9 % | 113× |
| *Référence : BTC perp buy & hold* | *10,0 %* | *52,2 %* | *0,44* | | *−78,0 %* | |
| *Référence : panier long équipondéré* | *−1,0 %* | *68,4 %* | *0,33* | | *−79,3 %* | |

Par année civile : 2021 S2 +10,5 % · 2022 −17,4 % · 2023 +5,0 % · 2024 +24,6 % · 2025 +22,0 % ·
2026 à date −6,3 %. Attribution annualisée : prix +12,9 %, funding +2,0 %, **coûts −7,4 %**.
Le portefeuille est à peu près market-neutral (exposition nette moyenne −0,01, brute 0,93,
corrélation journalière à BTC −0,11).

Ce que ces chiffres montrent, et ce qu'ils ne montrent pas :

* **Pas de meilleur rendement ajusté du risque que de détenir du BTC** (Sharpe 0,42 contre
  0,44). Les vraies différences sont un drawdown de −27 % au lieu de −78 % et un bêta proche de zéro.
* **Les edges synthétiques étaient optimistes.** Le carry de funding passe d'un Sharpe de 2,0
  sur le marché synthétique à 0,38 sur données réelles. Le changement de régime perd de l'argent
  dans les deux cas.
* **Les coûts prennent la moitié de l'edge brut.** Le framework les fait payer, c'est le but.
* Sur données réelles, le classement des alphas retient les signaux de funding
  (`funding_reversal_90` t = 3,4, `funding_carry_21` t = 2,0, tous deux significatifs après BH)
  et rejette le momentum 60 jours (t = 0,04). `low_vol_60` a l'IC de rang le plus élevé (t = 8)
  mais un Sharpe long-short **négatif**. L'IC pondère chaque jour de la même façon, alors que le
  PnL en dollars est dominé par les rares jours de forte dispersion où les alts volatils
  s'envolent. Un IC de rang n'est pas un PnL.
* **Limites.** L'univers est composé des survivants d'aujourd'hui, choisis a posteriori : c'est
  un biais du survivant en notre faveur. Il ne couvre qu'une plateforme, 12 actifs et une seule
  trajectoire, sans analyse de capacité. C'est un contrôle de cohérence de la chaîne sur données
  réelles, pas un historique de performance.

## Démarrage rapide

macOS / Linux :

```bash
git clone https://github.com/theojoyerclement-crypto/alphaforge.git alphaforge && cd alphaforge
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"                   # ou : pip install -r requirements-dev.txt
python scripts/smoke_test.py              # tests + backtest de référence + replay paper
```

Windows (PowerShell). Utiliser `python` et non `python3`, qui sous Windows est le raccourci du
Microsoft Store :

```powershell
git clone https://github.com/theojoyerclement-crypto/alphaforge.git alphaforge; cd alphaforge
python -m venv .venv; .venv\Scripts\Activate.ps1    # cmd.exe : .venv\Scripts\activate.bat
pip install -e ".[all]"
python scripts/smoke_test.py
```

`scripts/smoke_test.py` lance la suite de tests, puis `configs/multi_strategy.yaml`, puis un
replay paper-trading de 10 barres. Il affiche le chemin du rapport HTML
(`runs/<horodatage>_multi_strategy/report.html`) et sort en erreur au premier échec.
`--quick` saute les tests. Les commandes individuelles :

```bash
pytest                                                # 161 tests, ~2 min
alphaforge list                                       # stratégies, features, alphas
alphaforge research configs/multi_strategy.yaml       # classement des alphas
alphaforge run configs/multi_strategy.yaml            # backtest + rapport HTML dans runs/
alphaforge run configs/ml_walk_forward.yaml           # portefeuille ML en walk-forward
alphaforge run configs/multi_strategy.yaml --set portfolio.target_vol=0.15
alphaforge paper configs/multi_strategy.yaml --replay 20   # parité boucle paper / backtest
```

Sans installer le package : `PYTHONPATH=src python -m alphaforge ...`
(PowerShell : `$env:PYTHONPATH="src"; python -m alphaforge ...`).

### Données réelles (à lancer soi-même)

Nécessite `pip install ccxt` (inclus dans `.[all]`) et un accès réseau aux endpoints publics de
Binance. Aucune clé API n'est requise : seules des données de marché publiques sont téléchargées.

```bash
alphaforge fetch --exchange binance --timeframe 1d --start 2021-01-01 --symbols \
  BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT BNB/USDT:USDT XRP/USDT:USDT ADA/USDT:USDT \
  DOGE/USDT:USDT AVAX/USDT:USDT LINK/USDT:USDT DOT/USDT:USDT LTC/USDT:USDT ATOM/USDT:USDT
alphaforge run configs/binance_perps.yaml --set data.end=2026-10-12      # le tableau ci-dessus
alphaforge research configs/binance_perps.yaml --set data.end=2026-10-12
alphaforge run configs/binance_perps.yaml                                # jusqu'à la dernière barre clôturée
```

Le téléchargement prend environ 1 minute. Ensuite, `data/store/ccxt_binance_swap/<symbole>/`
contient `1d.ohlcv.parquet` et `events.funding.parquet`, chacun avec un fichier JSON qui
enregistre le nombre de lignes, la période et un hash du contenu. Chaque dossier de run
`runs/<horodatage>_binance_perps/` contient `report.html`, `metrics.csv`,
`pnl_attribution.csv`, `cleaning_report.csv` et un `manifest.json` avec le hash des données.
Après le premier téléchargement, le run lit le stockage local. Ajouter
`--set data.source=parquet` garantit qu'il reste hors ligne. Binance n'est pas accessible
depuis toutes les juridictions.

### Paper trading

```bash
alphaforge paper configs/multi_strategy.yaml --replay 20    # démo hors ligne + contrôle de parité
alphaforge paper configs/binance_perps.yaml                 # un pas live sur données fraîches
```

`PaperTrader` (`src/alphaforge/live/`) exécute les mêmes stratégies, le même
`PortfolioConstructor` et le même contrôle du drawdown que le backtest, sur les données jusqu'à
la dernière barre **clôturée**. Il trade ensuite, via un `Broker`, la cible décidée il y a
`execution_delay` barres.

* `--replay N` parcourt les N dernières barres une par une. À chaque barre, il ne voit que le
  passé, comme en live, et échoue si la NAV obtenue diffère de celle du `BacktestEngine` sur les
  mêmes barres.
* Le mode par défaut fait un pas par appel, à planifier une fois par barre (cron ou
  Planificateur de tâches). Il rafraîchit les données, valorise le compte, trade, puis écrit dans
  `paper/<expérience>/{state.json, fills.csv, equity.csv}`. Un second appel sur la même barre ne
  fait rien.
* `PaperBroker` exécute à la clôture avec le modèle de coûts du backtest. Un vrai connecteur de
  plateforme implémenterait la même interface `Broker` à trois méthodes (`sync`, `execute`,
  `nav/weights`), sans rien changer en amont. Les stratégies qui apprennent ne sont pas encore
  supportées, car elles nécessitent des modèles persistés et versionnés.

## Architecture

| Module | Rôle |
|---|---|
| `data` | Ingestion CCXT (OHLCV + funding, pagination, retries, bougie en cours écartée), générateur synthétique, nettoyage audité, stockage Parquet avec métadonnées hashées, panels `MarketData` alignés |
| `features` | 21 features causales : momentum, estimateurs de volatilité, RSI, efficiency ratio ; proxies de microstructure (Amihud, Roll, Corwin-Schultz, λ de Kyle) ; funding et basis |
| `alpha` | Bibliothèque de 15 alphas, IC de rang et décroissance, spreads par quantile, Newey-West, PSR/DSR, Benjamini-Hochberg, combinaison causale pondérée par l'IC |
| `strategies` | Momentum TS/XS, mean reversion, retournement court terme, régimes, carry, composite d'alphas, ML directionnel, meta-labeling, régimes GMM |
| `portfolio` | Risque EWMA, sizing inverse-vol, ciblage de volatilité ex-ante, risk parity ERC, pénalité de corrélation, limites, contrôle du drawdown, Kelly fractionnaire |
| `backtest` | Moteur événementiel, modèles de coûts linéaire et racine carrée, walk-forward, purged K-fold et CPCV |
| `ml` | Labels à horizon fixe et triple barrière, modèle directionnel poolé, meta-labeling avec bet sizing, régimes GMM et de volatilité |
| `analytics` | Sharpe, Sortino, Calmar, profit factor, VaR/CVaR, épisodes de drawdown, table mensuelle, graphiques, rapport HTML autonome |
| `research` | `run_experiment` / `run_alpha_research` : config → dossier de résultats avec manifeste |
| `live` | `PaperTrader`, interface `Broker` et `PaperBroker` : le code de recherche pilote un compte barre par barre, avec un état qui survit aux redémarrages |

Le dépôt et le package Python s'appellent tous deux `alphaforge`. Le nom du dossier local n'a
pas d'importance. Détails de conception : [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Méthodologie et contrôles de biais : [docs/METHODOLOGY.md](docs/METHODOLOGY.md).
Ajouter une stratégie : [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence et avertissement

[MIT](LICENSE). Logiciel de recherche : rien ici ne constitue un conseil en investissement. Les
résultats sur données synthétiques illustrent la méthodologie. Les résultats sur données réelles
proviennent d'un unique backtest, biaisé par le choix de survivants, et ne constituent pas un
historique de performance.
