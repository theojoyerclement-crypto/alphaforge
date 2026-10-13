# Guide d'étude AlphaForge

Fiches de révision pour préparer les entretiens, une par module. Chaque fiche contient : ce que
fait le module (avec une analogie), les fichiers et fonctions à connaître, les choix de
conception et les alternatives écartées, puis cinq questions d'entretien probables avec une
réponse modèle courte.

---

## Module 1 : performance du moteur de backtest (noyau numba)

### Ce que fait le module, en 5 phrases

1. Le moteur de backtest rejoue l'histoire barre par barre : chaque jour, il valorise le
   portefeuille, applique le funding, fait dériver les poids, puis paie les coûts pour revenir
   aux poids cibles.
2. Cette boucle ne peut pas être vectorisée d'un bloc, car chaque barre dépend de la précédente
   (poids dérivés, NAV, overlay de drawdown, impact qui dépend de la NAV).
3. Le profilage a montré que la lenteur ne venait pas des calculs, mais du coût d'appel de
   petites fonctions numpy répétées 20 000 fois (surtout `np.nan_to_num` dans le modèle de
   coûts).
4. Le module ajoute une copie compilée de la boucle avec numba, qui donne les mêmes résultats à
   1e-10 près et va de 4 à 32 fois plus vite selon la taille du panel.
5. La boucle Python reste la référence lisible et le repli automatique : le noyau n'est qu'un
   « jumeau rapide » vérifié par des tests.

**Analogie.** La boucle Python est un caissier qui, pour chaque article, appelle un collègue au
téléphone pour vérifier le prix : chaque vérification est rapide, mais le temps passe dans les
appels. Le noyau numba, c'est le caissier qui a appris la grille des prix par cœur : il fait
exactement le même travail, sans décrocher le téléphone.

### Les 3 fichiers et fonctions clés

| Fichier | Fonction | À retenir |
|---|---|---|
| `src/alphaforge/backtest/engine.py` | `BacktestEngine.run`, `_run_python`, `_use_kernel`, `_run_kernel` | `run` prépare des tableaux numpy, puis choisit la boucle. `_run_python` est la référence. `_use_kernel` applique les règles : numba installé, modèle de coûts et contrôleur intégrés. |
| `src/alphaforge/backtest/_kernel.py` | `_loop`, `_cost`, `_controller_update` | Réécriture de la boucle sur des tableaux simples, compilée avec `@njit(cache=True)`. `_controller_update` reproduit la machine à états du contrôleur de drawdown. |
| `scripts/benchmark.py` + `tests/test_kernel.py` | `measure`, `profile` ; `test_kernel_matches_python_reference` | Le benchmark mesure le temps médian et le pic mémoire, `--profile` donne le profil cProfile. Le test de parité est property-based (hypothesis) : toutes les séries doivent être égales à 1e-10. |

### Choix de conception et alternatives écartées

- **Mesurer avant d'optimiser.** cProfile (déterministe, il instrumente chaque appel) pour les
  pourcentages, py-spy (échantillonnage) pour la vue d'ensemble. Sans profil, on aurait
  probablement optimisé `mark_to_market`, qui ne pèse que 9 %.
- **numba plutôt que Rust (PyO3/maturin).** Une fois la boucle compilée, elle ne représente plus
  que 16 % d'un run. Rust gagnerait au mieux environ 1,2 fois de plus, au prix d'un second
  langage, d'une compilation en CI et de wheels par plateforme. Écarté.
- **Vectoriser davantage dans la boucle : écarté.** Le problème est le nombre d'appels par barre,
  pas leur contenu ; vectoriser ne réduit pas le nombre d'itérations.
- **Garder la boucle Python comme référence, pas la remplacer.** Elle reste lisible, auditée,
  utilisée par le paper trading, et sert d'oracle au test de parité. Un modèle de coûts
  personnalisé (hors des deux modèles intégrés) passe automatiquement par elle.
- **Tolérance 1e-10, pas l'égalité exacte.** numba additionne dans un ordre différent du produit
  scalaire BLAS de numpy, d'où des écarts de l'ordre de 1e-16 par barre. 1e-10 laisse une marge
  confortable tout en attrapant la moindre erreur de logique.
- **numba optionnel (`pip install "alphaforge[fast]"`).** Pas de dépendance lourde imposée ; la
  CI l'installe pour que le test de parité s'exécute vraiment.

### 5 questions d'entretien probables

**1. Pourquoi ne pas vectoriser toute la boucle avec numpy ?**
Parce qu'elle est dépendante du chemin : les poids dérivés, la NAV, l'impact et l'overlay de
drawdown à la barre *t* dépendent du résultat de la barre *t − 1*. On vectorise ce qui peut
l'être (volatilité, ADV, rendements, calculés une seule fois avant la boucle) et on compile le
reste.

**2. Comment avez-vous trouvé le goulot d'étranglement ?**
Avec cProfile sur un benchmark reproductible (données et cibles avec seed) : le modèle de coûts
représentait 71 % du temps, dont 44 % rien que pour `np.nan_to_num`. Un indice confirmait le
diagnostic : le temps augmentait surtout avec le nombre de barres et peu avec le nombre
d'actifs, ce qui est la signature d'un surcoût par itération, pas d'un coût de calcul.

**3. Comment garantissez-vous que la version rapide donne les mêmes résultats ?**
Par un test property-based (hypothesis) qui génère des cas aléatoires : taille, cotations
manquantes, délai, modèle de coûts, overlay, levier, ruine du portefeuille. Toutes les séries
(rendements, coûts, positions, equity, equity du book fantôme) et l'état final du contrôleur
doivent égaler la boucle Python à 1e-10. Je l'ai aussi fait tourner sur 1 500 cas en local.

**4. Pourquoi 1e-10 et pas une égalité exacte ?**
L'arithmétique flottante n'est pas associative : numba somme les produits dans un autre ordre
que le produit scalaire BLAS de numpy, ce qui donne des écarts de l'ordre de l'epsilon machine
(environ 1e-16) qui se cumulent sur quelques milliers de barres. 1e-10 reste six ordres de
grandeur au-dessus de ce bruit et bien en dessous de tout écart dû à une vraie erreur.

**5. Quel est le prochain goulot, et que feriez-vous ?**
Avec la boucle compilée, la préparation pandas (`cost_inputs` : volatilité EWMA et ADV
glissante) représente 58 % d'un run à 200 actifs. Pour des balayages (capacité, coûts), je
mettrais ces entrées de risque en cache entre les runs qui partagent les mêmes données, plutôt
que de réécrire la boucle en Rust.
