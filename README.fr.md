# AlphaForge

[![CI](https://github.com/theojoyerclement-crypto/alphaforge/actions/workflows/ci.yml/badge.svg)](https://github.com/theojoyerclement-crypto/alphaforge/actions/workflows/ci.yml) ![python](https://img.shields.io/badge/python-3.11%2B-1f4e79) ![license](https://img.shields.io/badge/license-MIT-555555)

**[Ouvrir un rapport HTML d'exemple](https://theojoyerclement-crypto.github.io/alphaforge/)** · [Résultats (en anglais)](docs/RESULTS.md) · [Méthodologie](docs/METHODOLOGY.md) · [Architecture](docs/ARCHITECTURE.md) · [Utilisation](docs/USAGE.md) · 🇬🇧 [English](README.md)

AlphaForge est un framework de recherche quant crypto conçu pour répondre à *« cet edge est-il
réel, et survit-il à l'exécution ? »* plutôt qu'à *« combien aurait-il rapporté ? »*. Il emmène
les perpétuels Binance des features causales jusqu'à une boucle de paper trading qui exécute le
code même du backtest, en passant par une recherche d'alphas qui contrôle les tests multiples,
des backtests qui paient coûts et funding, et du ML en walk-forward. Chaque chiffre publié sort
d'une config avec seed, les Sharpe de portefeuille ont des intervalles de confiance, et les
changements de méthode sont pré-enregistrés dans git avant le calcul de leurs résultats.

![Book point-in-time contre BTC buy & hold](docs/img/equity_pit_vs_btc.png)

## Trois conclusions

**1. Un alpha statistiquement significatif n'est pas une stratégie.** Un alpha de retournement
à 1 barre avec t = 5,5 affiche un Sharpe de +1,83 sans friction, −0,28 après frais, spread et
impact en racine carrée, et −2,91 une fois ajoutée une barre de latence.

![Stress des coûts](docs/img/cost_stress.png)

**2. Sur données réelles, rien ici ne bat BTC de façon fiable.** Le book Binance point-in-time
(chaque mois, les 12 perps les plus liquides du moment, un coin délisté inclus) a un Sharpe de
−0,08 [−0,89 ; 0,73] contre 0,44 [−0,39 ; 1,28] pour BTC buy & hold. Aucun Sharpe sur données
réelles n'exclut zéro à 95 % ; seul le book synthétique, où les edges ont été implantés, y
parvient.

![Sharpe avec intervalles à 95 %](docs/img/sharpe_intervals.png)

**3. Des contrôles honnêtes ont trouvé de vrais bugs, et les corrections sont publiées avec leur
impact.**
- Un solveur de risk parity qui divergeait en silence : Sharpe du portefeuille synthétique
  0,65 → 1,37 une fois corrigé.
- Binance qui sert des bougies plates fictives après un délisting : désormais nettoyées comme un
  flux mort.
- Un overlay de drawdown qui piégeait le book à faible exposition : corrigé par une règle
  [pré-enregistrée](docs/METHODOLOGY.md#6-pre-registered-change-drawdown-overlay-v2-2026-10-13)
  (exposition moyenne 0,24 → 0,89), sans prétendre qu'elle ajoute du Sharpe.

Tableaux complets, limites et commandes de reproduction : **[docs/RESULTS.md](docs/RESULTS.md)**
(en anglais).

## Par où commencer

| À regarder | Pourquoi c'est important | Où |
|---|---|---|
| **Une seule commande** | Tests, backtest de référence et replay paper, hors ligne, sur tout OS | `python scripts/smoke_test.py` |
| **Tests anti look-ahead** | On brouille le futur ; features, stratégies et construction de portefeuille doivent donner le même passé | `tests/test_features.py`, `tests/test_strategies.py`, `tests/test_portfolio.py` |
| **Comptabilité** | `net = brut + funding − coûts`, vérifié contre des carnets calculés à la main | `tests/test_backtest.py` |
| **Parité backtest ↔ paper** | La boucle paper recalcule chaque décision sur l'historique tronqué et retrouve l'equity du backtest à 1e-10 | `src/alphaforge/live/`, `tests/test_live.py` |
| **Données réelles, mauvais résultats inclus** | Univers fixe contre point-in-time, overlay v1 contre v2, référence BTC | [RESULTS §5](docs/RESULTS.md#5-real-market-binance-usdt-m-perpetuals) |
| **Incertitude** | Intervalles par bootstrap par blocs apparié sur chaque Sharpe de portefeuille | [RESULTS : bruit](docs/RESULTS.md#how-much-of-this-is-noise) |
| **Pré-registration** | Règle commitée avant les résultats (`cc2ba29` précède `7a968bc`) | [METHODOLOGY §6](docs/METHODOLOGY.md#6-pre-registered-change-drawdown-overlay-v2-2026-10-13) |
| **Exécution sur testnet** | Le même code de décision envoie de vrais ordres à un exchange sandbox : identifiants client idempotents, réconciliation, kill switch | [docs/LIVE_TESTNET.md](docs/LIVE_TESTNET.md), `src/alphaforge/live/ccxt_broker.py` |
| **Performance du moteur** | Profilé, puis un noyau numba optionnel testé en parité contre la boucle Python | [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |
| **Qualité des données** | Trous, flux morts, couverture du funding, délistings | `alphaforge validate`, `src/alphaforge/data/quality.py` |
| **La doc ne contredit jamais les résultats** | Un chiffre du README ou d'une figure absent de RESULTS.md fait échouer la CI | `tests/test_docs.py` |

## Démarrage rapide

macOS / Linux :

```bash
git clone https://github.com/theojoyerclement-crypto/alphaforge.git alphaforge && cd alphaforge
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
python scripts/smoke_test.py              # tests + backtest de référence + replay paper
```

Windows (PowerShell). Utiliser `python` et non `python3`, qui est le raccourci du Microsoft Store :

```powershell
git clone https://github.com/theojoyerclement-crypto/alphaforge.git alphaforge; cd alphaforge
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -e ".[all]"
python scripts/smoke_test.py
```

Le smoke test affiche le chemin du rapport HTML (`runs/<horodatage>_multi_strategy/report.html`).
Commandes individuelles :

```bash
alphaforge list                                         # stratégies, features, alphas
alphaforge research configs/multi_strategy.yaml         # classement des alphas (BH, DSR)
alphaforge run configs/multi_strategy.yaml              # backtest + rapport HTML dans runs/
alphaforge run configs/multi_strategy.yaml --set costs.fee_bps=2
alphaforge paper configs/multi_strategy.yaml --replay 20     # parité boucle paper / backtest
alphaforge validate configs/binance_perps_pit.yaml           # audit qualité (données réelles)
python scripts/make_figures.py                          # régénère les figures ci-dessus
```

Les runs sur données réelles nécessitent les données publiques de Binance (sans clé API) ; les
commandes de téléchargement sont dans [RESULTS.md](docs/RESULTS.md#reproducing-the-real-data-tables).

## Organisation du dépôt

```
src/alphaforge/   data · features · alpha · strategies · portfolio · backtest · ml · analytics · research · live · config · utils
configs/          multi_strategy · ml_walk_forward · cost_stress · binance_perps · binance_perps_pit
docs/             RESULTS · METHODOLOGY · ARCHITECTURE · USAGE · PERFORMANCE · LIVE_TESTNET · img/
scripts/          smoke_test.py · make_figures.py · benchmark.py · build_site.py
examples/         démarrage · recherche d'alphas · ML meta-labeling · stratégie maison · confiance du Sharpe
tests/            propriétés de causalité, comptabilité calculée à la main, fuites de CV, parité paper, cohérence de la doc
```

Fonctionnement, configuration, paper trading et écriture d'une stratégie :
[docs/USAGE.md](docs/USAGE.md) et [CONTRIBUTING.md](CONTRIBUTING.md) (en anglais).

## Feuille de route

- Balayages de capacité et de sensibilité aux coûts, désormais peu coûteux grâce au moteur compilé
- Une première session testnet réelle, journalisée et présentée à côté du backtest
- Échantillons réels plus longs ou plus larges : cinq ans de données journalières ne suffisent pas à départager les books testés ici
- Historique des instruments délistés depuis une seconde source, pour supprimer le biais du survivant restant

## Licence et avertissement

[MIT](LICENSE). Logiciel de recherche, pas un conseil en investissement. Les résultats sur données
réelles proviennent d'un unique backtest au biais du survivant réduit, pas d'un historique de
performance.
