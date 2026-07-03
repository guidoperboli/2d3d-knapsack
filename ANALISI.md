# Analisi sperimentale: GASP, GRASP-Parreño e ALNS per il container loading

Questo documento raccoglie in forma organica i risultati sperimentali
ottenuti sul pacchetto, pensato come base per la stesura del paper. Tutti
i numeri provengono da esecuzioni riproducibili incluse in `examples/`;
le statistiche seguono il protocollo onesto adottato nel progetto (media
su seed fissati a priori, mai best-of, salvo dove esplicitamente
indicato come riferimento e non per il confronto).

> **Stato dei risultati — da leggere prima di citare i numeri.**
> Ogni tabella è etichettata con il tipo di evidenza:
>
> - **[CAMPIONE]** — misura su poche istanze e/o un solo seed, ottenuta
>   in fase di sviluppo entro vincoli di tempo. Indicativa della
>   direzione e dell'ordine di grandezza, **non** definitiva: da
>   sostituire con la campagna completa prima della sottomissione.
> - **[CAMPAGNA]** — misura sull'intera classe (100 istanze) con seed
>   fissi multipli e statistica completa (media/std/min/max). Definitiva,
>   citabile nel paper.
> - **[RIFERIMENTO]** — valore dalla letteratura (Parreño et al. 2008),
>   riportato per confronto, non prodotto da noi.
>
> Allo stato attuale la maggior parte delle tabelle delle sezioni 4–6 è
> **[CAMPIONE]**: la direzione è solida e ripetuta, ma i valori esatti
> vanno consolidati lanciando `examples/run_overnight.py` (vedi §8).
> La tabella di fedeltà §2 su BR1 è già vicina al definitivo (5 istanze a
> 5000 iterazioni complete) ma va estesa alle 100 istanze di classe.

## 1. Dataset e protocollo

Le istanze sono le 1500 di Bischoff-Ratcliff / Davies-Bischoff (BR1–BR15,
100 per classe, da 3 a 100 tipi di scatole), lo stesso set completo usato
da Parreño et al. (2008) e dalla letteratura sul container loading.
Provenienza verificata: BR1–BR7 dagli originali OR-Library, BR8–BR15
dall'estensione Davies-Bischoff; allineamento controllato a livello di
token numerici (byte-identico ai nostri file sulle classi comuni).

Protocollo BR: i file riportano vincoli di orientamento verticale per
ogni scatola; la letteratura li rispetta. Il nostro protocollo usa sempre
`--respect-orientation` insieme al seme costruttivo e alla ricerca
locale. Le statistiche riportano media/deviazione standard/min/max sui
seed fissati; il "best-of" è segnalato come non confrontabile.

## 2. Fedeltà al GRASP di Parreño (a parità di iterazioni)

Abbiamo reimplementato fedelmente il GRASP di Parreño et al. 2008
(`gasp/grasp_parreno.py`): costruzione randomizzata con RCL al 100·δ%,
δ reattivo su {0.1..0.9} con periodo 500 e α=10, improvement per
rimozione del 50% finale dei blocchi e ricostruzione deterministica due
volte. Una iterazione = una costruzione randomizzata + una fase di
improvement, esattamente come nel paper, con criterio di stop a numero di
iterazioni alternativo al tempo.

**Risultato di fedeltà (BR1, 5000 iterazioni vere):** *[CAMPIONE — 5
istanze a 5000 iter; da estendere alle 100 della classe per il dato
[CAMPAGNA]]*

| | fill medio | tempo |
|---|---|---|
| Nostra reimplementazione (Python) | 92,66 % | ~31 s/istanza |
| Parreño (Tab. 4, C++ 2008) *[RIFERIMENTO]* | 92,95 % | ~8 s/istanza |
| **scarto** | **−0,29 punti** | — |

A parità di iterazioni la nostra reimplementazione riproduce il risultato
pubblicato entro mezzo punto. Su BR7 il fill cresce monotonicamente con
le iterazioni verso il valore di riferimento (91,62 %). Questo dimostra
che il divario a parità di **tempo** (nostro ~88 % vs loro ~92 % a 10 s)
non è algoritmico ma di **throughput** (Python vs C++): a parità di
iterazioni i metodi coincidono.

**Sul costo per iterazione.** Il tempo cresce fortemente con
l'eterogeneità, coerentemente con l'implementazione originale (Tab. 3 del
paper, dove i tempi vanno da ~1 s su BR1 a ~387 s su BR15 a 50000 iter).
Misure nostre (Python + kernel numba per la dominanza): BR1 ~6 ms/iter,
BR7 ~45 ms/iter, BR15 ~445 ms/iter. Le 5000 iterazioni complete sono
quindi raggiungibili sulle classi a bassa-media eterogeneità; sulle
classi dure il confronto si basa sul trend di convergenza.

## 3. Ottimizzazioni di throughput

Le ottimizzazioni hanno seguito sempre la profilazione, non l'intuizione.

- **Memoizzazione delle rotazioni**: `Item.rotations()` era chiamata
  ~145.000 volte per run e ricalcolava sempre lo stesso risultato; con
  cache, eliminata dai colli di bottiglia.
- **Distanza near-corner diretta**: `_near_corner` calcolava la distanza
  lessicografica con un ciclo 8×8 (1,2 M chiamate ad `abs`); riscritta
  con la formula diretta `min(coord, dim−coord)`. Costruttivo da 6 a 58
  costruzioni/secondo (~10×).
- **Kernel numba per la dominanza**: l'eliminazione degli spazi dominati
  (O(n²)) era il 58 % del tempo su BR15; compilata con numba
  (`gasp/ems_numba.py`), verificata identica al Python. BR15 da ~1660 ms
  a ~445 ms per costruzione (3,7×), BR7 da ~100 ms a ~45 ms. Le 5000
  iterazioni su BR15 scendono da ~2,3 h a ~37 min per istanza.

## 4. ALNS: un solver autonomo per il container loading

`gasp/alns.py` è un solver ALNS completo e autonomo, contributo distinto
da GASP: non usa il loop GASP, gli score o le policy, ma solo le
primitive geometriche condivise (maximal spaces, piazzamento a blocchi,
distanza lessicografica).

**Architettura.** La soluzione è una lista di blocchi Parreño (non
scatole singole). Sei operatori di distruzione pesati adattivamente
(random, worst, region, related/Shaw, segment, radial); ricostruzione
col costruttivo a blocchi; accettazione simulated-annealing con reheat;
selezione operatori a roulette con aggiornamento pesi Ropke-Pisinger.

**Risultato (BR7, 10 s, Python):** *[CAMPIONE — 6 istanze, 1 seed; da consolidare su 100 istanze e seed multipli]*

| metodo | fill medio |
|---|---|
| **ALNS** | **90,6 %** |
| GASP + seme + layout | 88,55 % |
| GRASP Parreño 5000 iter (riferimento, minuti) | 91,62 % |

L'ALNS supera il GASP di **+2,07 punti su ogni istanza** a parità di
tempo e in Python, avvicinandosi a ~1,2 punti dal GRASP di Parreño ma in
10 secondi anziché minuti. I pesi appresi premiano gli operatori "worst"
e "related".

**Nota sul budget.** A 10 s la ricostruzione regret-2 costa troppo per
iterazione (dimezza i giri) e non si ripaga: è fuori dal pool di default,
riattivabile per budget lunghi. La lezione, ricorrente nel progetto: a
budget di tempo stretto il throughput per iterazione conta più della
sofisticazione degli operatori.

## 5. ALNS profit-aware

L'obiettivo dell'ALNS è selezionabile (`objective_metric`): volume (CLP,
default) o profit (knapsack). In modalità profit, la selezione dei
blocchi, l'accettazione e il best-incumbent passano al profit totale; sul
container loading profit e volume coincidono, quindi le due modalità si
sovrappongono.

**Risultato sul knapsack okp (molteplicità ~3 copie/tipo):** *[CAMPIONE — 5 istanze okp, 1 seed]*

| | gap medio dall'ottimo |
|---|---|
| ALNS profit-aware | +1,13 % |
| GASP | +1,32 % |

Dove c'è molteplicità e profit ≠ volume, l'ALNS profit-aware è
competitivo col GASP e lo supera su diverse istanze (es. okp2 e okp5
raggiungono l'ottimo dove il GASP perde +1,8 % e +4,0 %). Dove la
molteplicità manca (ngcut, ~2 copie), volume e profit danno lo stesso
gap, perché i blocchi non si formano: lì il discriminante non è
l'obiettivo ma la molteplicità.

## 6. Ricostruzione ibrida blocchi/EP a scelta adattiva

Il pool di ricostruzione include sia la ricostruzione a blocchi
(costruttivo Parreño) sia una ricostruzione EP a scatole singole
(`_rebuild_ep`). I pesi adattivi scelgono l'una o l'altra per-istanza,
senza soglia di molteplicità a priori.

**Risultato (knapsack profit, bassa molteplicità):** *[CAMPIONE — istanze ngcut singole, 1 seed]*

| istanza | solo blocchi | blocchi + EP |
|---|---|---|
| ngcut01 | +4,9 % | **+0,0 %** (ottimo) |
| ngcut03 | +6,9 % | **+0,0 %** (ottimo) |

L'aggiunta dell'EP-repair porta all'ottimo proprio dove i blocchi
degeneravano. Sul container loading ad alta molteplicità (BR7) il fill
resta ~89 %, perché i pesi continuano a preferire i blocchi: la modifica
è additiva e retrocompatibile.

**Comportamento dei pesi e tuning.** Tarando i parametri di adattamento
(`react`, `seg_update`) è emerso un punto concettuale: i pesi separano
nettamente i due operatori **dove divergono davvero** (alta molteplicità,
es. BR1: EP ~0,6 contro blocchi ~1,1), mentre restano piatti **dove i due
sono equivalenti** (bassa molteplicità: con 1–2 copie un "blocco" è una
scatola singola, quindi block ed EP producono lo stesso piazzamento). I
pesi piatti, lì, sono la risposta corretta, non un difetto. Aggiornamenti
più frequenti (`seg_update`=25) con reazione moderata (`react`=0,1) danno
separazione più netta di una reaction-factor elevata, migliorando
leggermente anche il fill.

## 7. Principio unificante

Un filo conduttore attraversa tutti i risultati: **i blocchi rendono dove
c'è molteplicità e svaniscono dove non c'è.** Lo abbiamo osservato sul
costruttivo a blocchi, sul seme Parreño, sull'ALNS e sulla ricostruzione
ibrida. Da qui la complementarità dei metodi:

- **GASP** — solver generale (2D/3D, profit/volume, con o senza
  molteplicità); vince sul 2D-KP profit dove la molteplicità è assente.
- **GRASP-Parreño fedele** — reimplementazione validata; replica i
  risultati pubblicati a parità di iterazioni.
- **ALNS** — specialista del container loading; con ricostruzione ibrida
  adattiva e obiettivo selezionabile copre sia il volume (BR, dove batte
  GASP di +2 punti) sia il profit con molteplicità (okp, dove eguaglia o
  supera GASP), e degrada con grazia verso il comportamento EP dove la
  molteplicità manca.

Il discriminante metodologico non è "volume vs profit" né "2D vs 3D", ma
la **presenza o assenza di molteplicità**; la selezione adattiva degli
operatori dell'ALNS scopre da sé quale rappresentazione conviene su
ciascuna istanza, e la nettezza dei pesi finali misura quanto i due
regimi divergono.

## 8. Da campione a campagna: checklist di consolidamento

Per portare le tabelle da **[CAMPIONE]** a **[CAMPAGNA]** prima della
sottomissione, lanciare le campagne complete e aggiornare le tabelle
corrispondenti con media/std/min/max su seed fissi.

Una **singola notte di calcolo** consolida ora tutte le tabelle tranne
la fedeltà GRASP (separata perché molto pesante):

```
python examples/run_overnight.py --workers 8
```

produce le fasi: `br_all` (GASP volume, BR1-15), `br_alns` (ALNS volume,
BR1-15), `kp_gasp` (GASP profit, ngcut/okp/gcut), `kp_alns` (ALNS
profit-aware + ibrido EP, ngcut/okp/gcut), e il workbook con fogli
separati per solver.

| Tabella | Sezione | Fase/comando | Foglio xlsx |
|---|---|---|---|
| Fedeltà BR1/BR7/BR15 | §2 | `fidelity_run.py --workers 8` (separato) | — |
| ALNS vs GASP, volume | §4 | `br_all` + `br_alns` | "thpackN" vs "thpackN (ALNS)" |
| ALNS profit-aware, okp | §5 | `kp_gasp` + `kp_alns` | "okp" vs "okp (ALNS)" |
| Ibrido blocchi/EP, ngcut | §6 | `kp_gasp` + `kp_alns` | "ngcut" vs "ngcut (ALNS)" |

Dopo la notte, il workbook `results/Risultati_notte.xlsx` contiene i
fogli per solver con la statistica completa; i valori medi vanno
trasferiti nelle tabelle di questo documento sostituendo l'etichetta
[CAMPIONE] con [CAMPAGNA] e indicando il numero di istanze e di seed
effettivi. La fedeltà GRASP (§2) si lancia a parte con
`fidelity_run.py` perché le 5000 iterazioni sulle classi eterogenee
richiedono ore. Finché una tabella resta [CAMPIONE], nel testo del paper
va trattata come evidenza preliminare.

## 9. Riproducibilità

- Fedeltà GRASP: `python examples/fidelity_run.py --workers 8`
- Campagna GASP/ALNS su tutte le BR: `python examples/run_campaign.py
  --sets br --time 10 --respect-orientation [--solver alns] --workers 8`
- Notte di calcolo completa (GASP + ALNS + xlsx):
  `python examples/run_overnight.py --workers 8`

Tutte le statistiche di confronto usano la media sui seed fissati; il
workbook prodotto separa i fogli per solver (es. "thpack7" per GASP,
"thpack7 (ALNS)" per ALNS) per il confronto diretto classe per classe.
