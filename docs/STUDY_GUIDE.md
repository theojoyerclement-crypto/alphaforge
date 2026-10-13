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

---

## Module 2 : exécution sur testnet (CcxtTestnetBroker)

### Ce que fait le module, en 5 phrases

1. Le module fait passer les décisions du paper trader (les mêmes stratégies, la même
   construction de portefeuille et le même overlay que le backtest) de fills simulés à de vrais
   ordres sur le testnet d'un exchange.
2. Il ne change rien en amont : il implémente la même interface `Broker` que le broker simulé,
   avec `sync` pour lire le compte et `execute` pour envoyer les ordres.
3. Chaque ordre porte un identifiant client calculé à partir de (expérience, barre, symbole,
   sens) ; avant d'envoyer, et après toute erreur réseau, le broker cherche cet identifiant sur
   l'exchange, donc un ordre n'est jamais envoyé deux fois.
4. À chaque barre, il compare les positions réelles de l'exchange aux positions qu'il attend,
   et un kill switch arrête tout en cas de perte maximale, de désynchronisation ou de données
   périmées.
5. Toutes ses décisions sont écrites dans un journal JSON (une ligne par événement), et l'arrêt
   du kill switch survit à un redémarrage jusqu'à une réinitialisation manuelle.

**Analogie.** C'est un pilote de ligne avec une checklist. Avant chaque manœuvre, il vérifie
les instruments (données récentes, carburant suffisant, position de l'avion conforme au plan).
S'il n'a pas entendu la réponse de la tour, il ne répète pas son ordre à l'aveugle : il demande
d'abord si le premier a été reçu. Et la boîte noire enregistre tout.

### Les 3 fichiers et fonctions clés

| Fichier | Fonction | À retenir |
|---|---|---|
| `src/alphaforge/live/ccxt_broker.py` | `CcxtTestnetBroker.sync`, `execute`, `_place`, `_reconcile`, `client_order_id` | `_place` : recherche par identifiant client avant l'envoi, puis après chaque erreur réseau ; `DuplicateOrderId` traité comme « déjà placé ». `_reconcile` met à jour les positions attendues à partir des quantités réellement exécutées et les compare à l'exchange. |
| `src/alphaforge/live/kill_switch.py` | `KillSwitch.check_loss`, `check_positions`, `check_data`, `reset` | Trois déclencheurs indépendants. L'état « halted » est persisté avant de lever l'exception, et seul un humain le lève. |
| `src/alphaforge/live/retry.py` + `journal.py` | `call_with_retries`, `EventJournal.log` | Retries avec backoff exponentiel plafonné, uniquement sur les erreurs réseau. Journal en JSON lines, lisible après redémarrage. |

### Choix de conception et alternatives écartées

- **Identifiant client déterministe plutôt qu'aléatoire.** Un UUID aléatoire change à chaque
  tentative : après un crash, impossible de savoir si l'ordre était parti. Le hash
  (expérience, barre, symbole, sens) redonne le même identifiant, et la quantité est exclue
  exprès pour qu'un recalcul légèrement différent tombe sur le même ordre.
- **Chercher avant de réessayer, plutôt que réessayer directement.** Un timeout ne dit pas si
  l'exchange a reçu l'ordre. Réessayer à l'aveugle peut doubler la position ; on vérifie
  d'abord.
- **Réessayer seulement les erreurs réseau.** Un ordre refusé (marge insuffisante, quantité
  invalide) échouera à l'identique ; le réessayer masquerait le problème. Il est journalisé et
  le trading continue sur les autres symboles.
- **Positions attendues issues des fills, pas copiées de l'exchange.** Si on recopiait les
  positions de l'exchange à chaque fois, la réconciliation ne détecterait jamais rien (trade
  manuel, fill perdu).
- **Testnet uniquement, clés dans l'environnement.** Le broker refuse un exchange hors sandbox ;
  aucune voie vers la production dans le dépôt, aucune clé dans le code.
- **Exchange simulé en mémoire pour les tests**, plutôt que des appels au vrai testnet en CI :
  tests rapides, déterministes, capables de reproduire les pannes (timeout après acceptation,
  latence de réplication des positions), sans réseau ni clé.

### 5 questions d'entretien probables

**1. Comment garantissez-vous qu'un ordre n'est pas envoyé deux fois ?**
Par un identifiant client déterministe : hash de (expérience, barre, symbole, sens). Avant
d'envoyer, je cherche cet identifiant sur l'exchange. Si une requête finit en timeout, je
cherche de nouveau avant de réessayer, car l'exchange a pu l'accepter. Un `DuplicateOrderId`
est traité comme « déjà placé ». Les tests couvrent le timeout après acceptation et la reprise
après un crash.

**2. Que se passe-t-il si le programme plante entre l'envoi de l'ordre et la sauvegarde de
l'état ?**
Au redémarrage, la même barre est rejouée et produit les mêmes identifiants client. Les ordres
déjà passés sont retrouvés et non renvoyés. Les positions attendues sont recalculées à partir
des quantités exécutées trouvées, puis réconciliées avec l'exchange à la barre suivante.

**3. Pourquoi un kill switch qui ne se réarme pas tout seul ?**
Parce que la condition qui l'a déclenché (bug, trade manuel, marché anormal) n'a pas disparu
juste parce que le processus a redémarré. Un humain doit lire le journal, comprendre, puis
réinitialiser. L'état est persisté avant de lever l'exception : un test a d'ailleurs détecté
que ce n'était pas le cas dans la première version.

**4. Comment testez-vous du code d'exécution sans toucher à un exchange ?**
Avec un exchange simulé en mémoire qui reproduit l'interface CCXT : précision des quantités,
notional minimum, identifiants uniques, marge et PnL. Il sait injecter des pannes précises
(erreur réseau avant ou après acceptation, rejet, latence des positions). Les classes
d'erreurs sont injectées, donc la CI n'installe même pas ccxt.

**5. Quelles sont les limites de ce broker ?**
Ordres au marché uniquement, sans algorithme d'exécution ; le testnet n'a ni la liquidité ni
les prix du vrai marché, donc les fills ne disent rien des coûts réels ; le book fantôme de
l'overlay est simulé avec le modèle de coûts du backtest. Et aucune session réelle sur le
testnet n'a encore été faite : elle demande des clés et n'est pas dans la CI.
