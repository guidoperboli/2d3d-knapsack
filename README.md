# GASP — Greedy Adaptive Search Procedure for Multi-Dimensional Knapsack Problems

Python implementation of the metaheuristic described in:

> G. Perboli, *An Efficient Metaheuristic for Multi-Dimensional Knapsack Problems*.

The framework solves the **2D-KP** (with and without item rotation), the **3D-KP**
(with rotations) and the **3D-CLP** (Container Loading, profits equal to volumes),
separating the *feasibility phase* (an Extreme Point greedy, EP-KPH) from the
*optimality phase* (the item ordering, driven by adaptive scores with learning,
short- and long-term memory).

Pure Python, no external dependencies.

## Documento di analisi

I risultati sperimentali completi (fedelta' GRASP a parita' di
iterazioni, ottimizzazioni di throughput, ALNS e sue varianti
profit-aware e ibrida blocchi/EP, tuning, principio unificante sulla
molteplicita') sono raccolti in **ANALISI.md**, pensato come base per il
paper.

## Project layout

```
gasp-knapsack/
├── gasp/
│   ├── geometry.py        # Item, Placement, Knapsack, Packing (2D = 3D with H=1)
│   ├── extreme_points.py  # EP generation + Residual Space (Crainic et al. 2008)
│   ├── merit.py           # EP evaluation criteria: RS, MP, LEV, FF   (Sec. 3.1)
│   ├── sorting.py         # 8 sorting rules of the PCH                (Sec. 3.2)
│   ├── greedy.py          # EP-KPH greedy heuristic                   (Sec. 3.1)
│   ├── gasp.py            # GASP main loop, scoring, reinit           (Sec. 3.3-3.6)
│   ├── instances.py       # EP-style random generators + 1D-KP upper bound (Sec. 4.1)
│   ├── readers.py         # parsers for the bundled OR-Library benchmark sets
│   └── best_known.py      # proven optima / best-known values from the literature
├── data/                  # benchmark instances (OR-Library via ESICUP/datasets)
│   ├── 2d/  ngcut (12), cgcut (3), gcut (13), okp (5), hccut (5),
│   │        ngcutap/ngcutcon (21+21), ngcutfs1-3 (630, up to 4000 items)
│   └── 3d/  thpack1-7 = BR1-BR7 container loading (700 instances)
├── examples/run_demo.py        # random-instance demo
├── examples/run_benchmark.py   # benchmark runner with gaps vs optima
└── tests/test_gasp.py          # feasibility, parser, and optimality tests
```

## Quick start

```bash
python tests/test_gasp.py             # sanity + parser + optimality tests
python examples/run_demo.py 10        # random-instance demo
python examples/run_benchmark.py ngcut 10        # classic Beasley set vs optima
python examples/run_benchmark.py gcut 10         # gcut1-13
python examples/run_benchmark.py thpack1 10 20   # BR1, first 20 instances
```

## Benchmark instances and reference values

The `data/` folder bundles the OR-Library sets used in Section 4 of the
paper, taken from the ESICUP datasets mirror
(https://github.com/ESICUP/datasets):

| Set | Files | # inst. | Problem | Reference values |
|---|---|---|---|---|
| ngcut01-12 | `2d/ngcut/ngcutap.txt` (probs 1-12) | 12 | 2D-KP | proven optima in `best_known.py` |
| cgcut1-3 | `2d/cgcut/` | 3 | 2D-KP | proven optima |
| gcut1-13 | `2d/gcut/` | 13 | 2D-KP | optima for 1-12; gcut13 best-known only |
| okp1-5 | `2d/okp/` | 5 | 2D-KP | optima (Fekete-Schepers) |
| hccut1-5 | `2d/hccut/` | 5 | 2D-KP | optima in HC (1995), not bundled |
| ngcutap / ngcutcon | `2d/ngcut/` | 21+21 | 2D-KP | mixed (see file readme) |
| ngcutfs1-3 | `2d/ngcutfs/` | 630 | 2D-KP large | 1D-KP upper bound (Table 1 setup) |
| thpack1-7 (BR1-7) | `3d/thpack/` | 700 | 3D-CLP | mean fill rates (Table 4 refs) |

Loading is one call: `load_set("gcut")` returns `(name, items, knapsack)`
tuples; `optimum(name)` returns the proven/best-known value when bundled.
Constrained pieces (max Q copies) are expanded into Q individual items,
as customary in the 2D-KP literature. Entries flagged `verify=True` in
`best_known.py` are widely cited values that should be double-checked
against the original tables before publication use.

Not bundled (formats/parsers ready to extend): wang20 (value kept in
`best_known.py`), the Egeblad-Pisinger 2D/3D-KP rotation sets (use the
generators in `instances.py`, same parameterisation), and BR8-BR15
(strongly heterogeneous CLP, distributed by Davies & Bischoff).

Validation: GASP solves ngcut01-12 to the proven optimum (12/12) within
seconds.

## Performance backends and adaptive variant

- `gasp/fast_greedy.py`: numba-JIT EP-KPH kernel (auto-selected when
  numba is installed; disable with GASP_NO_NUMBA=1). Bit-identical to
  the pure-Python greedy; ~2-4x more iterations on the small 2D sets
  and ~200x on the BR container-loading instances.
- `gasp/adaptive.py` + `gasp/policies.py`: AdaptiveGASP replaces the
  fixed Score Update with a portfolio of targeted policies (pair swap,
  block swap, frontier, density band, waste matching) chosen by an
  ALNS-style roulette, plus prefix-cached warm starts of the greedy.
  At 30 s/instance it solves cgcut 3/3, okp 4/5 and gcut 6/12 to the
  proven optimum (mean gap ~0.5% on the 21 classic 2D instances).

Programmatic use:

```python
from gasp import GASP, GASPParams, generate_3d, knapsack_upper_bound

items, knapsack = generate_3d(n=40, geom_class="D", p=90, seed=11)
params = GASPParams(time_limit=10.0, allow_rotation=True, seed=1)
result = GASP(items, knapsack, params).run()

print(result.best_profit, result.iterations)
ub = knapsack_upper_bound(items, knapsack)
print("UB gap: %.2f%%" % (100 * (ub - result.best_profit) / ub))
```

A custom instance is just a list of `Item(idx, w, d, h, profit)` plus a
`Knapsack(W, D, H)` (use `h = H = 1` for 2D problems).

## Algorithm mapping to the paper

| Paper | Code |
|---|---|
| EP-KPH greedy (Sec. 3.1) | `greedy.ep_kph` |
| Merit functions FF / MP / LEV / RS | `merit.merit_value` |
| Extreme Points & Residual Space | `extreme_points.EPManager` |
| PCH initial solution, 8 sorting rules (Sec. 3.2) | `gasp.GASP.initial_solution`, `sorting.SORTING_RULES` |
| Score Initialization, s = k·p (k=3) (Sec. 3.3) | `GASP.init_scores` |
| Score Update with f^l, f^u, α=β=0.1, score swap (Sec. 3.4) | `GASP.update_scores` |
| Long-term Score Reinitialization + merit cycling RS→MP→LEV→FF (Sec. 3.5) | `GASP.long_term_reinit` |
| Parameter Update of k (Sec. 3.6) | inside `GASP.run` / `long_term_reinit` |
| Stopping: optimum reached or 10 s limit | `GASPParams.time_limit`, `known_optimum` |

## Notes and deviations

- **Instances.** The original benchmark files (ORLIB, Egeblad–Pisinger, BR1–BR15)
  are not redistributed here; `instances.py` provides generators following the
  same parameterisation (classes S/C, D, L, U; clustered/random; p ∈ {50, 90}).
  To reproduce the paper's tables, plug in a parser for the original files —
  any list of `Item`s plus a `Knapsack` works.
- **Score Update merit.** The paper defines the willingness measure as
  p/(w·l); this is kept verbatim (base area) also in 3D. Switching to volume
  is a one-line change in `update_scores` if desired.
- **Non-improving limit and reinit swaps** are not specified numerically in the
  paper; defaults (30 iterations, 5 swaps) are exposed in `GASPParams`.
- The 1D knapsack **upper bound** of Tables 2–3 is solved exactly by DP when the
  capacity is tractable, otherwise the Dantzig LP bound is used (still valid).

## Complexity

EP-KPH runs in O(n³) with the implemented merit functions, consistent with
Theorem 1 and Lemma 2 of the paper.

## Experimental: subset-based score policies

`GASPParams.update_policy` selects how scores are perturbed each iteration:
`classic` (the single (j,l) pair swap of Sec. 3.4), `band` (core-style
perturbation restricted to the critical band around the loaded/unloaded
frontier, à la Balas-Zemel/Pisinger core problem; size via `band_fraction`),
`waste` (spatial policy promoting unloaded items that fit the largest
residual EP free boxes), and `adaptive` (ALNS-style roulette over the three,
rewarded on best-solution improvements). Single-run comparison at 10 s on
the 14 hard classic instances: mean gap classic 2.27%, band 1.65%,
adaptive 2.30%; `band` closes cgcut2 and cgcut3 to proven optimality.


## Esecuzione locale ed installazione

Requisiti: Python 3.10-3.12 (testato su 3.12). Installazione
raccomandata in un ambiente virtuale dedicato, per evitare conflitti
con i pacchetti gia' presenti nel sistema (es. matplotlib/gensim che
richiedono numpy < 2).

Aprire una PowerShell Anaconda e creare un ambiente:

    python -m venv .venv

poi attivarlo, installare le dipendenze ed eseguire il collaudo rapido:

    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python examples/run_overnight.py --smoke --workers 2

In alternativa, su macOS / Linux l'attivazione e':

    source .venv/bin/activate

(Se PowerShell blocca l'attivazione: Set-ExecutionPolicy
-ExecutionPolicy RemoteSigned -Scope CurrentUser, una tantum; oppure
.venv\Scripts\activate.bat dal Prompt dei comandi. Con Anaconda:
conda create -n gasp python=3.12 e conda activate gasp.)

Il codice funziona sia con numpy 1.26.x sia con numpy 2.x.

`ortools` e' opzionale: serve solo per lo slave CP-SAT (`exact_repair`),
la sonda colonnare (`basin_probe`) e `gasp.cp_slave`; il core GASP e i
test girano con sole numpy+numba.

### Sequenza di comandi consigliata

Dopo l'installazione e il collaudo (`run_overnight.py --smoke`), questi
sono i comandi corretti e aggiornati per i casi d'uso principali. I
flag BR (`--respect-orientation --parreno-seed --layout-search`) vanno
SEMPRE usati insieme per i container loading: rispettano il protocollo
della letteratura e attivano le due estensioni che portano BR7 in fascia
alta (~88%).

    # 1. collaudo completo della catena (~1 minuto)
    python examples/run_overnight.py --smoke --workers 2

    # 2. notte di calcolo completa (tuning -> BR best-of-10 -> BR1-7 ->
    #    ngcutfs -> Excel), riprendibile; ~2-3 h con 8 worker
    python examples/run_overnight.py --workers 8

    # 3. solo BR7 best-of-10 col protocollo corretto (~20 min, 8 worker)
    python examples/run_campaign.py --sets thpack7 --seeds 10 --time 10 \
        --respect-orientation --parreno-seed --layout-search --workers 8

    # 4. tutte le classi BR1-7, single run, protocollo corretto
    python examples/run_campaign.py --sets thpack1 thpack2 thpack3 \
        thpack4 thpack5 thpack6 thpack7 --time 10 \
        --respect-orientation --parreno-seed --layout-search --workers 8

    # 5. le 630 ngcutfs (2D, niente flag BR) (~40 min, 8 worker)
    python examples/run_campaign.py --sets ngcutfs1 ngcutfs2 ngcutfs3 \
        --time 30 --workers 8

    # 6. workbook Excel da uno o piu' checkpoint
    python examples/make_results_xlsx.py --input results/campaign.json \
        --out results/Risultati.xlsx

    # 7. riepilogo testuale rapido di un checkpoint (anche parziale)
    python examples/run_campaign.py --summary results/campaign.json

Riferimento BR7 (protocollo vincolato della letteratura): metodi storici
80-88%, fascia alta ~88-90%, top-tier (VNS/GRASP maximal-space) ~91-92%.
Con i flag sopra il metodo raggiunge ~88% su BR7.


    python tests/test_gasp.py

oppure, con pytest installato:

    python -m pytest tests/ -v

La suite verifica: validita' geometrica dei packing (no sovrapposizioni,
no fuoriuscite, no duplicati) su tutti i criteri di merito e le regole
di ordinamento, equivalenza bit a bit tra backend Python e numba,
parser delle istanze, raggiungimento degli ottimi noti sulle istanze
piccole. Durata tipica: 2-4 minuti (la prima esecuzione include la
compilazione JIT dei kernel numba, ~30-60 s una tantum, poi in cache).

### Demo e benchmark

    python examples/run_demo.py            # singola istanza, output commentato
    python examples/run_benchmark.py       # campagna sui set classici

Su Windows: stessi comandi da PowerShell o dal terminale di VS Code
(numba e ortools hanno wheel binari per Windows; nessuna toolchain C
richiesta). Se la prima run numba fallisse per cache corrotta:
cancellare le cartelle `__pycache__` e riprovare.

### Campagne lunghe (run_campaign.py)

Per riprodurre le campagne complete del progetto in locale, con
checkpoint riprendibili (Ctrl+C in qualunque momento, poi rilanciare
lo stesso comando per continuare):

    python examples/run_campaign.py --list-sets
    python examples/run_campaign.py --sets thpack7 --seeds 10 --time 10
    python examples/run_campaign.py --sets ngcutfs1 ngcutfs2 ngcutfs3 --time 30
    python examples/run_campaign.py --sets ngcut cgcut gcut okp --seeds 5 \
        --time 30 --basin-probe --exact-repair
    python examples/run_campaign.py --summary results/campaign.json

La metrica e' scelta automaticamente: gap % vs ottimo provato dove
noto, gap % vs bound 1D esatto altrimenti, fill % per i set thpack
(CLP). Il riepilogo finale riporta single-run, best-of-N, media e
conteggio degli ottimi raggiunti per ogni set. Stime indicative:
best-of-10 su un set BR ~ 2h45; le 630 ngcutfs a 30 s ~ 5h15.

### Esecuzione parallela (multi-start)

`run_campaign.py` accetta `--workers N`: le run (istanza x seed) vengono
distribuite su N processi, rendendo il best-of-N quasi gratuito in
wall-time. Consigliato N = numero di core fisici; la prima run di ogni
processo carica i kernel numba dalla cache su disco. Funziona anche su
Windows (start method spawn). Esempio: best-of-10 su BR7 con 8 worker
~ 20 minuti invece di 2h45.

    python examples/run_campaign.py --sets thpack7 --seeds 10 --time 10 --workers 8

### Tuning sistematico (tune_params.py)

Random search sullo spazio dei parametri (alpha, beta,
non_improving_limit, reinit_swaps, k_init, pch_deltas), con la
configurazione di default sempre inclusa come baseline da battere.
Disegno appaiato: ogni configurazione gira sulle stesse celle
(istanza, seed); la classifica e' per rank medio per cella (robusto a
istanze di scala diversa), con il valore medio come spareggio. Il
vincitore viene salvato in un JSON direttamente usabile dal runner:

    python examples/tune_params.py --family br --configs 20 --seeds 2 --time 10 --workers 8
    python examples/run_campaign.py --sets thpack1 ... --config results/best_config.json

Preset di validazione: classic, br, ep3, mix (oppure --sets espliciti).
Checkpoint e ripresa come per le campagne. Suggerimento di metodo: fare
il tuning per famiglia (i parametri vincenti su BR possono non esserlo
sulle classiche) e validare la configurazione vincente su istanze NON
usate nel tuning prima di adottarla.

### La notte di calcolo (run_overnight.py)

Un solo comando per la sequenza completa: tuning BR -> tuning classic ->
best-of-10 su BR7 -> tutte le classi BR1-7 -> ngcutfs1-3 -> workbook
Excel finale. Le fasi BR usano il protocollo corretto (orientamento
vincolato, seme costruttivo di Parreno, ricerca sul layout). Log con
orari in results/overnight.log; ogni fase ha il proprio checkpoint,
quindi in caso di interruzione basta rilanciare lo stesso comando.

    python examples/run_overnight.py --workers 8
    python examples/run_overnight.py --smoke --workers 2     # collaudo ~1 min
    python examples/run_overnight.py --workers 8 --skip tune_br tune_classic

Fasi: tune_br, tune_classic, br7_bestof10, br_all, ngcutfs, xlsx (ognuna
saltabile con --skip). Durata indicativa con 8 worker: ~2h30-3h. Al
termine: results/best_br.json e best_classic.json (parametri),
br7_bestof10.json, br_all.json, ngcutfs.json (campagne),
Risultati_notte.xlsx.

### Excel dei risultati (make_results_xlsx.py)

Dal checkpoint di una campagna (anche parziale, anche piu' file fusi
insieme) si genera il workbook formattato:

    python examples/make_results_xlsx.py
    python examples/make_results_xlsx.py --input results/campaign.json \
        results/br_bestof10.json --out results/Risultati_GASP.xlsx

Un foglio per set (run per seed, best/mean e gap o fill come formule
vive, ottimi provati evidenziati in verde) piu' il foglio Riepilogo con
formule cross-foglio. Richiede openpyxl; le formule vengono ricalcolate
da Excel o LibreOffice all'apertura.

### Parametri di default

Dalla versione corrente i default di `GASPParams` sono la configurazione
**cfg03** selezionata dal tuning (`examples/tune_params.py`), prima
classificata sia sulla famiglia BR sia sulle classiche 2D:

    alpha=0.2  beta=0.1  k_init=2  non_improving_limit=10  reinit_swaps=3

Battono la baseline del paper (alpha=0.1, k_init=3, lim=30, swaps=5) di
~0.8 punti di fill su BR fuori campione, senza regressioni sugli ottimi
certificati. I valori originali dell'articolo restano nei commenti del
codice e sono riproducibili passandoli esplicitamente a GASPParams.

### Tempi di calcolo

Il tempo riportato (colonna "Tempo medio (s)" nell'Excel, "t/ist(s)" nel
summary) e' il tempo TOTALE di soluzione per istanza, al netto della
compilazione JIT di numba: i kernel sono precompilati una volta per
processo prima di ogni misura cronometrata. La ricerca locale finale e'
ritagliata dentro il budget complessivo, quindi il tempo totale resta
~time_limit (non time_limit + extra). Riferimenti di letteratura: 3D BR
fascia metaeuristica (Parreno) ~secondi/istanza, metodi pesanti ~decine
di minuti fino a >1h; 2D classico (gcut/ngcut) gli esatti chiudono
all'ottimo in <1s, quindi li' conta la qualita' (ottimi raggiunti), non
la velocita'.

### Protocollo statistico per il confronto con la letteratura

I metodi della letteratura sul container loading riportano la MEDIA su
piu' esecuzioni (5 o 10 seed), non il best-of. Per un confronto onesto e
a prova di obiezione, il runner riporta media, deviazione standard e
min/max sui seed FISSATI A PRIORI (1..N), MAI il best-of:

    python examples/run_campaign.py --sets thpack7 --seeds 10 --time 10 \
        --respect-orientation --parreno-seed --layout-search --workers 8
    python examples/run_campaign.py --summary results/campaign.json

Il riepilogo mostra 'mean', 'std', 'min', 'max' (statistiche oneste) e
'best-of' separato, marcato come NON usabile per il confronto (sarebbe
seed cherry-picking). Nell'Excel il foglio per set ha le stesse colonne,
con 'Mean fill %' come riferimento e 'Best-of (NO conf.)' segnalato. Su
BR7 il metodo mostra varianza bassa (spesso nulla: attrattore
deterministico), quindi media e best-of quasi coincidono e il numero e'
robusto al seed.

### Ricostruzione ibrida blocchi+EP (scelta adattiva)

Il pool di ricostruzione dell'ALNS include sia la ricostruzione a
BLOCCHI (costruttivo Parreno) sia una ricostruzione EP a SCATOLE SINGOLE.
I pesi adattivi scelgono l'uno o l'altro per-istanza, senza soglia di
molteplicita' a priori: dove la molteplicita' e' alta (BR) vincono i
blocchi, dove e' bassa (ngcut/okp) entra l'EP. Misura: aggiungere l'EP
porta ngcut01 e ngcut03 da +4.9%/+6.9% di gap (soli blocchi) a +0.0%
(ottimo); su BR7 il container loading resta ~89% perche' i pesi
continuano a preferire i blocchi. E' la sintesi delle due anime del
pacchetto in un solo solver adattivo.

### Integrazione ALNS nelle campagne

Il runner e l'orchestratore notturno supportano l'ALNS come solver
alternativo a GASP, via il flag --solver:

    python examples/run_campaign.py --sets br --time 10 \
        --respect-orientation --solver alns --workers 8 --out results/alns.json

L'ALNS e' autonomo: ignora i flag GASP-only (--parreno-seed,
--layout-search, che sono intrinseci al suo costruttivo) e usa i propri
parametri. La notte di calcolo (run_overnight.py) esegue ora due fasi
BR distinte, br_all (GASP) e br_alns (ALNS), su tutte le 15 classi; il
workbook finale ha fogli separati per solver ("thpack7" per GASP,
"thpack7 (ALNS)" per ALNS) cosi' il confronto e' diretto, classe per
classe, con la stessa statistica onesta (media/std/min/max sui seed).

### ALNS (Adaptive Large Neighborhood Search) - solver autonomo

Il modulo gasp/alns.py e' un solver ALNS COMPLETO e AUTONOMO per il
container loading, contributo distinto da GASP: non usa il loop GASP, gli
score, le policy o il portfolio adattivo. Condivide solo le primitive
geometriche (maximal spaces, piazzamento a blocchi, distanza
lessicografica), come e' normale tra solver dello stesso package.

Architettura: rappresentazione a BLOCCHI (lista di blocchi Parreno, non
scatole singole); 6 operatori di distruzione pesati adattivamente
(random, worst, region, related/Shaw, segment, radial); ricostruzione
col costruttivo a blocchi Parreno (best-volume/best-fit, piu' regret-2
disponibile per budget lunghi); accettazione simulated-annealing con
reheat; selezione operatori a roulette con aggiornamento pesi
Ropke-Pisinger. Massimizza il volume occupato (obiettivo CLP); per il
2D/3D knapsack profit usare GASP.

Risultato (BR7, 10s, Python): ALNS completa ~90.6% contro 88.55% del
GASP+seme+layout, +2.07 punti, su ogni istanza. I pesi appresi premiano
gli operatori "worst" e "related". Nota: a 10s il regret costa troppo
per iterazione (dimezza i giri), quindi e' fuori dal pool di default ma
riattivabile per budget lunghi. API: solve_alns(items, ks,
ALNSParams(time_limit=..., seed=...)) -> ALNSResult.

### GRASP-Parreno fedele e confronto a parita''' di iterazioni

Il modulo gasp/grasp_parreno.py e' una reimplementazione fedele del
GRASP di Parreno et al. 2008 (costruzione randomizzata con RCL al
100*delta%, delta reattivo su {0.1..0.9} con periodo 500 e alpha=10,
improvement per rimozione del 50% finale dei blocchi e ricostruzione
deterministica due volte). Una ITERAZIONE = una costruzione randomizzata
+ una fase di improvement, esattamente come nel paper, con criterio di
stop a numero di iterazioni (max_iter) alternativo al tempo. Serve per
il confronto a PARITA' DI ITERAZIONI con i risultati pubblicati, usato
solo per il container packing.

Risultato di fedelta' (examples/fidelity_grasp_parreno.py): su BR1 a
5000 iterazioni vere otteniamo ~92.5% di media contro il 92.95% di
Parreno (Tab.4) - entro mezzo punto. Su BR7 il fill cresce
monotonicamente con le iterazioni verso il loro 91.62%@5000. Questo
dimostra che il gap a parita' di TEMPO (nostro ~88% vs loro ~92% a 10s)
non e' algoritmico ma di throughput (Python vs C++): a parita' di
iterazioni i metodi coincidono. Il pipeline GASP (parreno_seed +
layout_search) resta l'alternativa pratica che da' gran parte del
riempimento in pochi secondi.

Accelerazione: l'eliminazione degli spazi dominati (il collo di bottiglia
su istanze eterogenee, ~58% del tempo) e' compilata con numba
(gasp/ems_numba.py), con fallback Python e warm-up per-processo. Guadagno
misurato: BR15 da ~1660ms a ~445ms per costruzione (3.7x), BR7 da ~100ms
a ~45ms; 5000 iterazioni su BR15 scendono da ~2.3h a ~37min per istanza.
Risultati invariati al centesimo (kernel verificato identico al Python).

### Istanze BR (Bischoff-Ratcliff / Davies-Bischoff)

Il pacchetto include l'INTERO set BR1-BR15 (thpack1-thpack15), 100
istanze per classe, 1500 in totale: lo stesso set completo usato da
Parreno et al. e dalla letteratura sul container loading. Le classi
vanno da 3 tipi di scatole (BR1) a 100 (BR15), coprendo da debolmente a
fortemente eterogeneo. BR1-BR7 sono gli originali Bischoff-Ratcliff 1995,
BR8-BR15 l'estensione Davies-Bischoff 1999. Provenienza dei dati estesi:
repository CLP-Datasets (kcliu2, mantenuto da Liu et al.), verificato a
livello di token numerici byte-identico ai nostri BR1-BR10 su tutte e 10
le classi comuni, a conferma dell'allineamento. L'alias di set "br"
carica tutte e 15 le classi (1500 istanze); per singola classe usare
"thpack1".."thpack15".

### Protocollo BR: vincoli di orientamento

Le istanze BR (thpack1-7) includono, per ogni tipo di scatola, dei flag
di orientamento verticale (xv/yv/zv): la letteratura sul container
loading li rispetta (orientamenti vincolati, stabilita), mentre la
rotazione libera risolve un problema diverso e i numeri non sono
confrontabili. Per misurare con lo stesso protocollo della letteratura:

    python examples/run_campaign.py --sets thpack7 --time 10 \
        --respect-orientation --layout-search --workers 8

Con i vincoli attivi, il riempimento medio su BR7 resta ~86-87% (la
metaeuristica recupera il vincolo che penalizza il solo costruttivo),
collocando il metodo nella fascia medio-alta della letteratura, sotto i
soli metodi top-tier (VNS/GRASP maximal-space, ~91-92% su BR7).

### Uso delle funzioni esatte

    from gasp import GASPParams
    from gasp.adaptive import AdaptiveGASP
    r = AdaptiveGASP(items, ks, GASPParams(time_limit=30,
                                           basin_probe=True,
                                           exact_repair=True)).run()
