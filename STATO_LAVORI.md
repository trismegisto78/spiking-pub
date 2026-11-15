# Spiking-Evo: stato dei lavori e come proseguire

Aggiornato al 14 novembre 2026. Questo documento serve a riprendere il lavoro da un'altra
macchina (il tuo PC) o in un'altra sessione senza dover ricostruire il contesto.

---

## 1. In breve

| cosa | stato |
|---|---|
| Milestone 1 (baseline) | **completo**, criterio di successo superato |
| Ottimizzazione per Runpod (CPU) | **completa**, verificata in locale con Docker |
| Milestone 2 (ambiente ricco) | da iniziare |
| Milestone 3 (diversità e comunicazione) | da iniziare |
| Milestone 4 (analisi) | da iniziare |
| Test | 81, tutti verdi; `ruff` pulito |
| Dipendenze runtime | solo `numpy` e `pyyaml` |

Dove sta il codice oggi:

- repository dedicato [`trismegisto78/spiking-evo`](https://github.com/trismegisto78/spiking-evo),
  branch `main`
- la cronologia è stata estratta da `trismegisto78/digital-projects` (cartella
  `spiking-evo/`, PR in bozza #7), che non è più la copia di riferimento

La specifica originale è in [`SPEC.md`](SPEC.md); il README spiega uso e scelte di progetto.

---

## 2. Risultati ottenuti

### Milestone 1: criterio "la fitness media cresce in modo misurabile in 100 generazioni"

Config di esempio della spec (`spiking_evo/experiments/food_only_baseline.yaml`): seed 42,
popolazione 200, griglia 40×40, 60 cibi con rigenerazione 0.01, 1000 passi per episodio.

| | generazione 0 | generazione 99 |
|---|---|---|
| fitness media | 19.9 | 515.8 |
| media prime / ultime 10 generazioni | 193 | 519 |
| agenti vivi a fine episodio | 73.5% | 100% |
| celle distinte visitate (media) | 81 | 689 |
| firing rate medio | 21.6 Hz | 23.6 Hz |

Controllo comportamentale fatto a mano: il genoma migliore della generazione 99, rivalutato su
20 episodi mai visti, raccoglie in media 540 di energia; con i sensori del cibo azzerati ne
raccoglie 8 e quasi non si muove (40 celle invece di 700). Il comportamento quindi dipende dai
sensori. Questo **non** dimostra ancora una ricerca mirata del cibo: per quello serve il test
comportamentale previsto dal milestone 2 (vedi sezione 7).

### Prestazioni (misurate su un container con 4 core)

| modalità | tempo per generazione |
|---|---|
| prima implementazione | 9.9 s |
| propagazione sparsa degli spike | 1.6 s |
| ottimizzazioni attuali, 1 processo | 1.3 s |
| 1 run con `--workers 4` | 0.84 s |
| `sweep` di 4 seed in parallelo | 0.47 s per run |

Un baseline da 300 generazioni dura circa 6.5 minuti su un core.

---

## 3. Mappa del codice

```
spiking-evo/                (radice del repository)
├── SPEC.md                 specifica originale
├── README.md               uso, scelte di progetto, sezione Runpod
├── STATO_LAVORI.md         questo file
├── pyproject.toml          pacchetto, dipendenze, config pytest e ruff
├── Dockerfile              immagine CPU per Runpod (python slim + sshd)
├── runpod/
│   ├── start.sh            entrypoint del pod: SSH, sweep da variabili d'ambiente
│   └── setup_on_pod.sh     installazione su un pod con immagine standard
└── spiking_evo/
    ├── cli.py              comandi: run, evaluate, sweep, cpus
    ├── seeding.py          derivazione dei sotto-seed dal seed globale
    ├── sweep.py            più run in parallelo, resumable, rilevamento CPU del container
    ├── config/             schema (dataclass) e loader YAML/TOML con validazione stretta
    ├── core/
    │   ├── world.py        griglia, ordine di aggiornamento, episodio, arene
    │   ├── agent.py        stato degli agenti (struttura di array)
    │   └── clock.py        tick del mondo vs passi della SNN
    ├── snn/
    │   ├── neuron.py       LIF con parametri per neurone
    │   ├── network.py      rete batch, ritardi interi, propagazione sparsa, ring buffer
    │   ├── encoders.py     rate / latency / current
    │   └── controller.py   osservazioni -> spike -> azioni motorie
    ├── stimuli/
    │   ├── base.py         protocollo Stimulus (punto di estensione)
    │   ├── registry.py     registrazione con @register e auto-discovery dei moduli
    │   └── food.py         FoodStimulus
    ├── genetics/
    │   ├── genome.py       genoma diretto, repair, serializzazione JSON versionata
    │   ├── operators.py    mutazione, crossover, selezione per torneo
    │   └── population.py   ciclo generazionale, valutazione parallela, checkpoint
    ├── fitness/evaluator.py   termini di fitness separati
    ├── metrics/logger.py      CSV, snapshot, checkpoint, config.yaml con commit hash
    ├── experiments/           file di config
    └── tests/                 81 test (pytest)
```

Non ancora creato: `genetics/speciation.py` (milestone 3).

---

## 4. Decisioni prese e scostamenti dalla spec

Tutte documentate anche nel README; qui le più importanti da ricordare prima di modificare il
codice.

1. **`Stimulus.sense` e `interact` lavorano sul batch di agenti**, non su un agente per
   chiamata come scritto nella spec. Una chiamata per agente costerebbe circa un milione di
   chiamate Python a generazione. La semantica è la stessa.
2. **Arene isolate per default** (`world.arenas: isolated`): ogni agente ha una copia privata
   del mondo, quindi la fitness non dipende dagli altri. `shared` (tutti nello stesso mondo,
   competizione per il cibo) è supportato ma non è parallelizzabile.
3. **Regola di riproducibilità per chi scrive stimoli:** ogni estrazione casuale diversa per
   arena va fatta con `world.per_arena(rng.<metodo>, ...)`. Così i risultati restano identici
   con qualsiasi numero di worker. Per ogni nuovo stimolo va aggiunto un caso in
   `tests/test_parallel.py`, che verifica proprio questa invarianza.
4. **LIF:** integrazione esatta dell'esponenziale, salti sinaptici istantanei, input ignorati
   durante la refrattarietà. Soglia e `tau_m` evolvono, `v_reset` e refrattarietà no.
5. **Ritardi:** interi in `[1, max_delay]`; uno spike emesso al passo `t` arriva a `t + d`.
6. **Motori:** quattro neuroni d'uscita in due coppie antagoniste (avanti/indietro,
   sinistra/destra), decodificate come differenza di rate sulla finestra di un tick.
7. **Mutazioni:** i tassi su pesi, soglie, `tau`, ritardi valgono per singolo gene; quelli
   strutturali (aggiungere/togliere sinapsi e neuroni) per genoma. `add_neuron` spezza una
   sinapsi (stile NEAT).
8. **Crossover** opzionale (default 0): allineato per indice di neurone e per coppia
   (sorgente, destinazione). È rozzo: per un crossover serio servono numeri di innovazione,
   da valutare insieme alla speciazione (milestone 3).
9. **Fitness:** `energy` = energia ottenuta dagli stimoli durante la vita;
   `firing_penalty` = `-w * max(0, rate - target) / target`; `silence_penalty` = `-w` se nessun
   neurone non di input spara mai; `exploration` = `w * celle distinte`.
10. **Attività contata solo da vivi:** firing rate e conteggio degli spike si fermano quando
    l'agente muore.
11. **Morte:** con energia a zero l'agente smette di muoversi e mangiare
    (`agent.die_at_zero`, disattivabile).

---

## 5. Come lavorare sul tuo PC

### Passo 1: clonare il repository

Il progetto vive nel repository dedicato `trismegisto78/spiking-evo` (fatto il 15/11/2026,
cronologia dei commit conservata).

### Passo 2: installazione (PowerShell su Windows, o terminale su macOS/Linux)

Serve Python 3.11 o superiore.

```powershell
git clone https://github.com/trismegisto78/spiking-evo.git
cd spiking-evo
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest                    # 81 test, pochi secondi
ruff check .
```

### Passo 3: primi esperimenti

```powershell
python -m spiking_evo cpus          # core disponibili
python -m spiking_evo run spiking_evo/experiments/food_only_baseline.yaml --generations 20
python -m spiking_evo run spiking_evo/experiments/food_only_baseline.yaml --workers 4
python -m spiking_evo sweep spiking_evo/experiments/food_only_baseline.yaml --seeds 1-4
python -m spiking_evo evaluate runs/food_only_baseline/42/snapshots/gen_0019.json `
    --config spiking_evo/experiments/food_only_baseline.yaml --episodes 20
```

Output in `runs/<esperimento>/<seed>/` (cartella ignorata da git): `generations.csv`,
`terms.csv`, `snapshots/`, `checkpoint.json`, `config.yaml`, e `run.log` per gli sweep.
Un run interrotto riprende con `--resume`; uno `sweep` rilanciato riprende da solo.

### Passo 4: Claude sul PC

Apri la cartella del progetto nell'app Claude Desktop, oppure lancia `claude` in un terminale
dentro la cartella. Con `claude remote-control` la sessione locale compare anche nell'app
Claude Code.

Nota: la cartella `spike` è già nella lista Remote Control del PC, ma oggi il PC non la offre
alle sessioni cloud. Di solito l'app Claude va aggiornata (su Windows serve una versione
recente) oppure la cartella sta in una directory nascosta. Anche dopo, l'accesso delle sessioni
cloud alle cartelle del PC potrebbe non essere ancora disponibile per il tuo account; una
sessione che gira direttamente sul PC funziona comunque.

### Cose non ancora verificate su Windows

Il codice è pensato per essere portabile (processi avviati con `spawn`, conteggio CPU con
ripiego per sistemi non Linux), ma è stato testato solo su Linux. Alla prima esecuzione sul PC
conviene lanciare `python -m pytest` e segnalare eventuali errori. `runpod/*.sh` e il
`Dockerfile` non servono in locale.

---

## 6. Runpod (quando servono più core)

- Usa **pod CPU**, non GPU: la simulazione è dominata dall'overhead per passo, non dal calcolo.
- Prezzi letti dal catalogo Runpod il 14/11/2026, per 32 vCPU: **cpu3c 0.96 $/ora**
  (Compute-Optimized, 9 data center), cpu5c 1.12 $/ora (CPU più recenti, solo EU-RO-1 ed
  EUR-IS-1). Fatturazione al secondo; per avviare un pod serve un credito che copra un'ora.
- **Stima** (non misurata su Runpod): 32 seed × 300 generazioni in 10–15 minuti, circa
  0.20–0.30 $. Il valore reale dipende da quanto rende un vCPU di Runpod rispetto a un core.
- Un benchmark per avere numeri reali costa meno di 0.10 $: 32 seed × 10 generazioni su un pod
  cpu3c da 32 vCPU. Non è ancora stato fatto.
- Procedura completa (immagine Docker, variabili `SWEEP_CONFIGS` / `SWEEP_SEEDS` /
  `SWEEP_ARGS`, network volume su `/workspace`) nella sezione Runpod del README.
- A fine lavoro **termina** il pod: uno fermato continua a pagare il disco.
- L'immagine va costruita e pubblicata su un tuo registry; non è ancora stata pubblicata.
  La build è stata verificata in locale (324 MB) simulando un pod con 2 CPU, volume montato e
  riavvio con resume.

---

## 7. Come proseguire

Ordine consigliato, coerente con la spec ("milestone 1 per intero prima di toccare il 2").

### Prima del milestone 2 (piccoli, utili subito)

1. **Eseguire i test sul PC** e sistemare eventuali problemi specifici di Windows.
2. **Run lungo di riferimento:** 300 generazioni e almeno 4–8 seed con `sweep`, per avere una
   baseline con varianza tra seed da confrontare con gli esperimenti successivi.
3. **Script di analisi minimo** (anticipa una parte del milestone 4): leggere
   `runs/*/*/generations.csv` e disegnare la fitness media e massima per generazione, con
   media e banda tra seed. Richiede `matplotlib`, da aggiungere come dipendenza opzionale
   (`[analysis]`) motivandola nel commit.

### Milestone 2: ambiente ricco

Criterio della spec: gli agenti distinguono zone di cibo e pericolo, misurato con un test
comportamentale.

1. **`stimuli/hazard.py` → `HazardStimulus`**: celle o zone che sottraggono energia. Canali
   sensoriali egocentrici come il cibo (si può riusare `sector_kernels` di `food.py`).
   `interact` ritorna un delta negativo.
2. **`stimuli/food_patch.py` → `FoodPatchStimulus`**: cibo raggruppato in zone (centri casuali,
   distribuzione gaussiana attorno ai centri), rigenerazione dentro la zona.
3. Regole da rispettare per entrambi:
   - nessuna modifica a `core/`: solo un file in `stimuli/` con `@register("...")`;
   - estrazioni casuali per arena tramite `world.per_arena(...)`;
   - test prima del codice: senso dei settori, effetto sull'energia, rigenerazione,
     invarianza al numero di worker (estendere `tests/test_parallel.py`).
4. **Nomi dei canali:** due stimoli dello stesso tipo richiedono `name` diversi nella config
   (es. `food` e `food2`); il registry lo verifica già.
5. **Test comportamentale:** comando o script che prende un genoma salvato e misura, su molti
   episodi, ad esempio:
   - tempo passato in zone di cibo vs zone di pericolo, confrontato con un agente casuale e
     con il genoma a sensori azzerati;
   - scelta in un ambiente a due bracci (cibo da una parte, pericolo dall'altra) con posizioni
     scambiate a caso.
   Probabilmente un nuovo sottocomando `behaviour` nella CLI.
6. **Config di esperimento** `experiments/food_hazard_patches.yaml`.

### Milestone 3: diversità e comunicazione

1. **Speciazione** (`genetics/speciation.py`): distanza genetica (sinapsi disgiunte e
   differenza dei pesi sulle coppie comuni), soglia di compatibilità, fitness condivisa.
   Riempire la colonna `n_species` di `generations.csv`, oggi vuota.
2. **Genoma indiretto:** regole di crescita a partire da un seme; selezione con
   `genetics.encoding: indirect` (oggi la config accetta solo `direct`).
3. **`SignalChannel`:** gli agenti emettono valori visibili ai vicini. Richiede
   `world.arenas: shared`, quindi niente parallelismo a blocchi; e richiede un canale motorio
   in più per "emettere". Attenzione: il numero di uscite oggi è fisso a 4
   (`snn/controller.py`, `N_MOTOR_OUTPUTS`). Rendere configurabili le uscite motorie è una
   modifica al core da documentare prima di farla, come chiede la spec.

### Milestone 4: analisi

Notebook o script che legge `runs/` e produce grafici di convergenza, mappe di attività (dove
si muovono gli agenti) e confronti tra esperimenti. Per le mappe serve salvare le traiettorie:
oggi non sono registrate, solo il conteggio delle celle visitate.

---

## 8. Limiti noti e debiti tecnici

- Un singolo run scala poco oltre 4 worker (overhead Python costante per passo). Se servirà
  molta più velocità per run, le opzioni sono Numba sul ciclo SNN o un backend alternativo
  (Brian2/Norse sono previsti come opzionali dalla spec, non implementati).
- Il crossover non usa numeri di innovazione (vedi punto 8 della sezione 4).
- Le uscite motorie sono fisse a 4.
- Il riepilogo della fitness non include ancora intervalli tra seed: arriva con lo script di
  analisi.
- Il repository non ha CI: conviene aggiungere un workflow GitHub Actions con `pytest` e `ruff`
  nel nuovo repository.

---

## 9. Messaggio pronto per una nuova sessione Claude

Da incollare all'inizio di una sessione sul PC:

> Lavoriamo su spiking-evo (simulazione evolutiva di agenti con reti spiking). Leggi
> `STATO_LAVORI.md`, `README.md` e `SPEC.md`. Prima esegui `python -m pytest` e `ruff check .`
> e dimmi se qualcosa fallisce su questa macchina. Poi proponi il piano per il milestone 2
> (`HazardStimulus`, `FoodPatchStimulus`, test comportamentale) rispettando le regole della
> sezione 7: nessuna modifica a `core/` per i nuovi stimoli, estrazioni casuali con
> `world.per_arena`, test prima del codice.
