# AlphaForge

**Framework de recherche alpha quantitative pour les marchés crypto.**
Ingestion des données → features causales → découverte d'alphas sous contrôle des tests
multiples → familles de stratégies → construction de portefeuille sous contrainte de risque →
backtest événementiel avec frictions réalistes → ML en walk-forward → paper trading sur le même
code → reporting. Le tout est piloté par un fichier de configuration typé et reproductible au
commit, à la seed et au hash des données près.

🇬🇧 [English version](README.md) · 📐 [Architecture](docs/ARCHITECTURE.md) · 🔬 [Méthodologie](docs/METHODOLOGY.md) · 🤝 [Contribuer](CONTRIBUTING.md)

[![CI](https://github.com/theojoyerclement-crypto/alphaforge/actions/workflows/ci.yml/badge.svg)](https://github.com/theojoyerclement-crypto/alphaforge/actions/workflows/ci.yml) ![python](https://img.shields.io/badge/python-3.11%2B-1f4e79) ![tests](https://img.shields.io/badge/tests-195%20passing-4f8a3c) ![license](https://img.shields.io/badge/license-MIT-555555) ![status](https://img.shields.io/badge/status-research-8064a2)

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
| **Incertitude assumée** | Chaque Sharpe de portefeuille a un intervalle par bootstrap par blocs apparié. Sur données réelles, aucun ne se distingue de 0 ni de BTC, et le README le dit | [Bruit](#quelle-part-de-bruit-) |
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
| **Portefeuille (risk parity, cible 20 % de vol)** | **30,3 %** | 21,0 % | **1,37** | 2,08 | −18,2 % | 168× |
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

> **Correction (audit de la Phase A).** Les lignes « portefeuille » des tableaux 3 et 5 avaient
> d'abord été publiées avec un solveur de risk parity qui **divergeait en silence** quand des
> sleeves étaient négativement corrélées. Il renvoyait par exemple 100 % sur une seule sleeve au
> lieu de contributions au risque égales, et s'écartait de l'ERC de plus de 25 points sur 18 %
> des barres de `multi_strategy` et 1 % de `binance_perps`. Le solveur est désormais un Newton
> amorti sur la formulation convexe de l'ERC, avec des contributions égales à 1e-12 et un test
> de régression sur le cas défaillant. Lignes corrigées : Sharpe synthétique 0,65 → **1,37**,
> car un vrai ERC diversifie mieux la sleeve carry. Sharpe Binance 0,42 → **0,35**, car les
> barres mal allouées étaient des paris concentrés qui avaient été chanceux. Les sleeves, la
> recherche d'alphas, le stress des coûts et le ML en walk-forward n'utilisent pas le solveur et
> sont inchangés (l'equity walk-forward bouge de moins de 1e-10). Les lignes « portefeuille » ont
> ensuite encore bougé avec l'overlay de drawdown v2 pré-enregistré : voir
> [Overlay de drawdown v1 contre v2](#overlay-de-drawdown-v1-contre-v2).

### 5. Marché réel : perpétuels Binance USDT-M

`configs/binance_perps.yaml` **n'a pas été modifié depuis le commit initial** et n'a donc pas été
réglé sur ces données. Univers : 12 grands perpétuels (BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX,
LINK, DOT, LTC, ATOM), barres journalières, funding réel toutes les 8 h, évaluation de juil. 2021
à oct. 2026 (données jusqu'au 2026-10-12), frictions par défaut (frais 4 bps, demi-spread
1,5 bps, impact racine carrée, 1 barre de latence).

| Net de tous coûts et du funding | CAGR | Vol | Sharpe | Sortino | Max DD | Turnover/an |
|---|---:|---:|---:|---:|---:|---:|
| **Portefeuille (risk parity, cible 20 % de vol)** | **9,0 %** | 20,4 % | **0,52** | 0,78 | −32,8 % | 141× |
| Momentum time-series | 5,5 % | 15,5 % | 0,43 | 0,61 | −22,9 % | 38× |
| Carry de funding | 5,1 % | 16,7 % | 0,38 | 0,53 | −28,8 % | 60× |
| Momentum cross-sectionnel | 2,1 % | 17,0 % | 0,21 | 0,29 | −30,7 % | 91× |
| Changement de régime | −11,5 % | 16,1 % | −0,68 | −0,93 | −49,9 % | 113× |
| *Référence : BTC perp buy & hold* | *10,0 %* | *52,2 %* | *0,44* | | *−78,0 %* | |
| *Référence : panier long équipondéré* | *−1,0 %* | *68,4 %* | *0,33* | | *−79,3 %* | |

Par année civile : 2021 S2 +10,5 % · 2022 −17,3 % · 2023 +24,7 % · 2024 +26,2 % · 2025 +21,9 % ·
2026 à date −9,9 %. Attribution annualisée : prix +17,1 %, funding +2,2 %, **coûts −8,6 %**.
Le portefeuille est à peu près market-neutral (exposition nette moyenne −0,02, brute 1,07,
corrélation journalière à BTC −0,10).

**5b. Le même portefeuille sur un univers point-in-time** (`configs/binance_perps_pit.yaml`).
La liste de 12 coins ci-dessus est un choix a posteriori : ces coins sont gros aujourd'hui
*parce qu'ils* ont survécu et monté. Cette variante garde les mêmes stratégies, frictions et
dates. Chaque mois, elle trade les 12 perps les plus liquides du moment, selon le volume en
dollars des 30 derniers jours et après 90 jours de cotation. Les 29 candidats incluent des coins
listés plus tard (APT, ARB, OP, SUI...) et **FTM, délisté en janvier 2025**. Binance continue de
servir des bougies FTM plates, à volume nul, après le délisting. Le nettoyage les écarte comme
un flux mort, si bien que FTM sort de l'univers au lieu de devenir un actif « sans risque » à
volatilité nulle.

| Net de tous coûts et du funding | CAGR | Vol | Sharpe | Max DD | Turnover/an |
|---|---:|---:|---:|---:|---:|
| **Portefeuille (overlay par défaut, v2)** | **−3,3 %** | 19,0 % | **−0,08** | −42,7 % | 111× |
| Portefeuille, overlay v1 (piégé) | −4,5 % | 6,9 % | −0,63 | −29,5 % | 25× |
| Portefeuille, sans overlay | −5,1 % | 20,6 % | −0,15 | −49,9 % | 123× |
| Momentum time-series | 5,0 % | 15,3 % | 0,39 | −20,4 % | 35× |
| Momentum cross-sectionnel | 1,1 % | 15,2 % | 0,15 | −21,8 % | 65× |
| Carry de funding | −5,7 % | 15,0 % | **−0,32** | −41,3 % | 47× |
| Changement de régime | −11,5 % | 15,7 % | −0,70 | −50,4 % | 102× |

Ce que ces chiffres montrent, et ce qu'ils ne montrent pas :

* **Le choix a posteriori de l'univers semble valoir environ 0,6–0,7 de Sharpe, mais ce
  n'est pas établi.** Overlay désactivé, le portefeuille passe de +0,53 sur l'univers a
  posteriori à −0,15 en point-in-time. L'intervalle apparié à 95 % de l'écart est
  [−0,05 ; 1,30]. Le carry de funding passe de +0,38 à −0,32, et seul le momentum time-series
  résiste (0,43 → 0,39). Le chiffre de référence est le chiffre point-in-time.
* **Rien ici ne bat la détention de BTC de façon fiable.** Sur l'univers flatteur, le Sharpe du
  portefeuille (0,52) est un peu au-dessus de celui de BTC (0,44), avec un drawdown de −33 % au
  lieu de −78 % et un bêta proche de zéro. L'intervalle apparié de l'écart est
  [−1,17 ; 1,31]. Sur l'univers point-in-time, le Sharpe est négatif. Voir
  [Quelle part de bruit ?](#quelle-part-de-bruit-).
* **Les coûts prennent environ la moitié de l'edge brut en prix.** Le framework les fait payer,
  c'est le but.
* Sur données réelles, le classement des alphas retient les signaux de funding
  (`funding_reversal_90` t = 3,4, `funding_carry_21` t = 2,0, tous deux significatifs après BH)
  et rejette le momentum 60 jours (t = 0,04). `low_vol_60` a l'IC de rang le plus élevé (t = 8)
  mais un Sharpe long-short **négatif**. L'IC pondère chaque jour de la même façon, alors que le
  PnL en dollars est dominé par les rares jours de forte dispersion où les alts volatils
  s'envolent. Un IC de rang n'est pas un PnL.
* **Limites.** Les coins que Binance ne sert plus du tout (LUNA, MATIC...) ne peuvent pas être
  candidats : le biais du survivant est réduit, pas supprimé. Il n'y a qu'une plateforme et une
  seule trajectoire, sans analyse de capacité. C'est un contrôle de cohérence de la chaîne sur
  données réelles, pas un historique de performance.

### Overlay de drawdown v1 contre v2

L'overlay v1 mesurait le drawdown sur l'equity qu'il produisait lui-même. Une fois l'exposition
réduite, cette equity cessait de bouger et le drawdown se figeait. Le portefeuille 5b est resté
à 5–10 % d'exposition pendant quatre ans. La correction a été **pré-enregistrée** dans
[METHODOLOGY §6](docs/METHODOLOGY.md#6-pre-registered-change-drawdown-overlay-v2-2026-10-13)
et commitée (`cc2ba29`) avant le calcul de tous les chiffres ci-dessous. La v2 mesure le
drawdown sur un **book fantôme** : les mêmes cibles à exposition pleine, avec les mêmes coûts et
le même funding. Les seuils sont inchangés (15 % / 30 % / 20 barres). La v1 reste disponible
via `risk.drawdown_basis: realized` et reproduit exactement les anciens tableaux.

| Sharpe du portefeuille (max DD) | v1 (realized) | **v2 (fantôme, défaut)** | sans overlay |
|---|---:|---:|---:|
| Synthétique multi-stratégies | 1,37 (−18,1 %) | **1,37 (−18,2 %)** | 1,38 (−17,8 %) |
| Binance, univers fixe | 0,35 (−27,7 %) | **0,52 (−32,8 %)** | 0,53 (−31,2 %) |
| Binance, point-in-time | −0,63 (−29,5 %) | **−0,08 (−42,7 %)** | −0,15 (−49,9 %) |
| Portefeuille ML walk-forward | −0,13 (−25,9 %) | **−0,05 (−29,7 %)** | 0,19 (−26,9 %) |
| Multiplicateur d'exposition moyen (PIT) | 0,24 | **0,89** | 1,00 |

Publié tel quel, sans réglage supplémentaire :

* **Le piège a disparu.** Dans le run point-in-time, l'exposition moyenne passe de 0,24 à 0,89,
  et la part des barres sous la moitié de l'exposition de 85 % à 11 %.
* **La v2 n'apporte pas de valeur démontrable par rapport à l'absence d'overlay.** Les écarts
  ne sont pas significatifs sur Binance, et la v2 fait significativement *moins bien* sur le
  portefeuille ML (−0,24 [−0,45 ; −0,03]). L'écart v2 − v1 n'est pas significatif non plus,
  même en point-in-time (+0,54 [−0,17 ; 1,25]).
* **Les faibles drawdowns de la v1 étaient un artefact.** Elle était hors du marché. La v2 ne
  plafonne **pas** le drawdown du portefeuille à la limite dure de 30 %, car cette limite
  s'applique au drawdown du book fantôme : le max DD atteint −33 % sur l'univers fixe et −43 %
  en point-in-time.
* Toute nouvelle modification de l'overlay nécessitera une nouvelle pré-registration.

### Quelle part de bruit ?

Intervalles à 95 % par bootstrap par blocs apparié, avec une méthode fixée à l'avance
([METHODOLOGY §7](docs/METHODOLOGY.md#7-confidence-intervals-on-published-sharpe-ratios)).
Pour reproduire : `python examples/05_sharpe_confidence.py`.

| Sharpe du portefeuille, intervalle à 95 % | v2 (défaut) | sans overlay | BTC buy & hold |
|---|---:|---:|---:|
| Synthétique multi-stratégies | **1,37 [0,60 ; 2,12]** | 1,38 [0,61 ; 2,13] | |
| Portefeuille ML walk-forward | −0,05 [−1,04 ; 0,99] | 0,19 [−0,82 ; 1,24] | |
| Binance, univers fixe | 0,52 [−0,28 ; 1,32] | 0,53 [−0,25 ; 1,34] | 0,44 [−0,39 ; 1,28] |
| Binance, point-in-time | −0,08 [−0,89 ; 0,73] | −0,15 [−0,96 ; 0,65] | 0,44 [−0,39 ; 1,28] |

| Écart de Sharpe apparié | Estimation [95 %] | Exclut 0 ? |
|---|---:|:---:|
| Binance fixe : v2 − BTC | +0,08 [−1,17 ; 1,31] | non |
| Binance PIT : v2 − BTC | −0,53 [−1,72 ; 0,65] | non |
| Binance v2 : univers fixe − point-in-time | +0,61 [−0,05 ; 1,30] | non (limite) |
| Binance fixe : v2 − v1 | +0,17 [−0,09 ; 0,44] | non |
| Binance PIT : v2 − v1 | +0,54 [−0,17 ; 1,25] | non |
| Binance fixe : v2 − sans overlay | −0,01 [−0,13 ; 0,12] | non |
| Binance PIT : v2 − sans overlay | +0,07 [−0,12 ; 0,27] | non |
| ML walk-forward : v2 − sans overlay | −0,24 [−0,45 ; −0,03] | **oui** |
| Synthétique : v2 − sans overlay | −0,01 [−0,02 ; −0,00] | **oui** (minime) |

Le verdict, sans détour :

* **Sur données réelles, aucun Sharpe de ce dépôt ne se distingue de zéro, ni de BTC.** Cinq ans
  de rendements journaliers donnent des intervalles d'environ ±0,8. Le seul Sharpe
  significativement positif est celui du marché synthétique, où les edges ont été implantés.
* **L'effet de l'univers choisi a posteriori est important mais non établi** (+0,61, un
  intervalle qui touche tout juste zéro).
* **Il n'est pas démontré que l'overlay v2 bat la v1.** Là où l'overlay diffère
  significativement de l'absence d'overlay, il fait *moins bien* (ML −0,24 ; synthétique
  −0,01). Son intérêt tient au fait qu'il ne piège plus le book, pas à un gain de Sharpe.

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
pytest                                                # 195 tests, ~2 min
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

# 5b, univers point-in-time : 17 candidats de plus, puis le run PIT
alphaforge fetch --exchange binance --timeframe 1d --start 2021-01-01 --end 2026-10-12 --symbols \
  NEAR/USDT:USDT FIL/USDT:USDT APT/USDT:USDT ARB/USDT:USDT OP/USDT:USDT SUI/USDT:USDT \
  INJ/USDT:USDT AAVE/USDT:USDT UNI/USDT:USDT ETC/USDT:USDT BCH/USDT:USDT TRX/USDT:USDT \
  SAND/USDT:USDT AXS/USDT:USDT ICP/USDT:USDT 1000SHIB/USDT:USDT FTM/USDT:USDT
alphaforge validate configs/binance_perps_pit.yaml --set data.end=2026-10-12   # drapeaux qualité
alphaforge run configs/binance_perps_pit.yaml --set data.end=2026-10-12
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
