# Spiking-Evo: stato dei lavori e come proseguire

Aggiornato al 14 novembre 2026. Questo documento serve a riprendere il lavoro da un'altra
macchina (il tuo PC) o in un'altra sessione senza dover ricostruire il contesto.

---

## 1. In breve

| cosa | stato |
|---|---|
| Milestone 1 (baseline) | **completo**, criterio di successo superato |
| Ottimizzazione per Runpod (CPU) | **completa**, verificata in locale con Docker |
| Milestone 2 (ambiente ricco) | **in corso**: pericoli a più livelli e terreno modificabile fatti; `FoodPatchStimulus` e test comportamentale formale da fare |
| Milestone 3 (diversità e comunicazione) | da iniziare |
| Milestone 4 (analisi) | da iniziare |
| Test | 120, tutti verdi; `ruff` pulito |
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

### Milestone 2, primo passo: pericoli su griglia 50×50 (15/11/2026)

Nuovo stimolo `hazard` (`stimuli/hazard.py`): zone circolari fisse che tolgono energia a ogni
passo a chi ci sta dentro; percepite con canali propri (fronte/sinistra/destra/qui), separati
da quelli del cibo. Esperimenti: `food_hazard_50.yaml` (danno 1) e `food_hazard_50_d3.yaml`
(danno 3), griglia 50×50, 95 cibi (stessa densità del baseline), 8 zone da raggio 2.5 (~7%
della griglia). 100 generazioni, seed 42, ~2 minuti con `--workers 4`.

Test comportamentale fatto a mano: genoma migliore della generazione 99 su 20 episodi nuovi;
"cieco ai pericoli" = stessa rete con i canali del pericolo azzerati.

| esperimento | fitness media gen 0 → 99 | tempo nei pericoli (evoluto) | tempo nei pericoli (cieco) | energia evoluto / cieco |
|---|---|---|---|---|
| danno 1 | 4.0 → 351.9 | 4.1% | 6.2% | 372 / 359 |
| danno 3 | → 225.9 | **1.2%** | 6.0% | **229 / 74** |

Con danno 1 il pericolo costa poco e l'evitamento è debole; con danno 3 gli agenti evitano le
zone (5 volte meno tempo dentro) e, se privati del senso del pericolo, perdono due terzi
dell'energia. È un singolo seed: va confermato con uno `sweep` su più seed.

### Pericoli con danni diversi e terreno modificabile (15/11/2026)

**Pericoli a più livelli.** `damage` può essere una lista, ad esempio `[1, 3, 6]`: ogni zona
riceve un livello a caso, e dove le zone si sovrappongono vale il più forte. Le creature
percepiscono il pericolo in proporzione al danno (`danno / danno massimo`), quindi una zona
più pericolosa "si sente" di più.

**Terreno modificabile** (`stimuli/terrain.py`). Ogni creatura ha un neurone d'uscita in più,
"agisci", che opera sulla cella dove si trova. Il terreno ha una qualità nascosta, che le
creature non percepiscono:
- su terreno fertile l'azione pianta un raccolto, che matura dopo `grow_time` passi e
  mangiato rende `harvest_energy` (conseguenza positiva, ritardata);
- su terreno sterile l'azione avvelena la cella per `poison_duration` passi, e chi ci sta
  sopra perde `poison_damage` a ogni passo (conseguenza negativa, immediata);
- ogni azione costa `action_cost`.
Le creature percepiscono soltanto le conseguenze (raccolti maturi, veleno), non il terreno.

**Unica modifica al core, e perché.** Per modificare l'ambiente serve un'uscita motoria
nuova; prima il core conosceva solo i 4 neuroni di movimento. La modifica è generica: il
parametro `network.action_outputs` aggiunge K neuroni d'azione, il controller li decodifica
(azione attiva sopra `action_threshold_hz`) e il mondo li mette in `agents.action` prima che
gli stimoli agiscano. Il core non sa cosa fa un'azione: lo decide lo stimolo che la legge.
Con `action_outputs: 0` (default) i risultati sono identici bit per bit a prima.

**Misure per stimolo.** Gli stimoli possono esporre `episode_stats()`; i contatori finiscono in
`stimuli.csv` (media e massimo per generazione): cibo mangiato, passi e danno nei pericoli,
azioni, semine, raccolti, avvelenamenti, passi sul veleno.

**Risultati** (seed 42, 150 generazioni, ~4 minuti con `--workers 4`; valori medi per creatura
per episodio):

| esperimento | gen 0 | gen 50 | gen 125 | gen 149 |
|---|---|---|---|---|
| `terrain_50` (20% fertile, maturazione 100 passi): fitness | -16.5 | 242 | 256 | 258 |
| azioni / semine / raccolti | 5.3 / 0.3 / 0 | 0.1 / 0 / 0 | 0.2 / 0 / 0 | 0 / 0 / 0 |
| `terrain_50_easy` (48% fertile, maturazione 30 passi, azione meno costosa): fitness | 54 | 534 | 2439 | 1138 |
| azioni / semine / raccolti | 81 / 11 / 6 | 277 / 72 / 43 | 661 / 258 / 178 | 354 / 133 / 87 |

- Con il terreno difficile l'evoluzione **elimina** l'azione: agire a caso avvelena quasi
  sempre, e il premio è troppo lontano nel tempo per essere scoperto.
- Con il terreno facile emerge l'**agricoltura**: le creature seminano e tornano a raccogliere,
  e quasi abbandonano il cibo selvatico (da 17 a circa 5 cibi per episodio). La fitness oscilla
  molto tra generazioni (ogni generazione è valutata su un episodio diverso): va confermata con
  più seed e `fitness.episodes` maggiore di 1.
- Domanda aperta e interessante: dove sta il punto di passaggio tra i due esiti (percentuale
  fertile, tempo di maturazione, costo)? È un buon esperimento da fare con `sweep`.

### Varietà di raccolto (15/11/2026)

`crops` in `terrain` definisce più varietà `{name, grow_time, energy, weight}`. Ogni zona
fertile ne coltiva una, estratta in proporzione a `weight`; la varietà fa parte del terreno
nascosto e si scopre solo quando il raccolto matura (da maturo è percepito in proporzione
all'energia). `stimuli.csv` conta i raccolti per varietà.

Esperimento `terrain_50_crops.yaml` (terreno come `terrain_50_easy`):

| varietà | maturazione | energia | peso |
|---|---|---|---|
| slow | 30 passi | 15 | 3 |
| medium | 15 passi | 7 | 3 |
| fast | 5 passi | 2 | 3 |
| jackpot | 5 passi | 60 | 0.5 (in circa metà degli episodi c'è una zona) |

Seed 42, 150 generazioni (valori medi per creatura per episodio):

| gen | fitness | vive | semine | avvelenamenti | raccolti slow / medium / fast / jackpot |
|---|---|---|---|---|---|
| 0 | 37 | 64% | 27 | 3.1 | 1.3 / 0 / 21.9 / 0 |
| 50 | 496 | 73% | 54 | 16.4 | 1.9 / 33.2 / 4.6 / 4.3 |
| 100 | 1288 | 93% | 270 | 7.3 | 18.0 / 132.0 / 93.9 / 0 |
| 125 | 948 | 98% | 488 | 1.3 | 0.4 / 1.1 / 478.8 / 0.5 |
| 149 | 827 | 71% | 360 | 18.7 | 1.2 / 10.8 / 321.1 / 3.0 |

Media fitness: 46 nelle prime 10 generazioni, 1062 nelle ultime 10.

- La strategia dominante è la coltura **rapida a ciclo continuo**: centinaia di semine e
  raccolti per episodio, ciascuno da 2 di energia. Rende più delle varietà lente perché il
  costo dell'azione è basso (0.05) e il ciclo è brevissimo.
- Quale varietà viene raccolta dipende molto dalla disposizione dell'episodio (le zone vicine
  al punto di partenza): le creature evolvono un comportamento generico ("semina dove sei e
  raccogli subito"), non una preferenza per una varietà.
- Il jackpot viene raccolto qualche volta (fino a 4 a episodio, cioè 260 di energia), ma non
  diventa un obiettivo: è troppo raro per essere cercato.
- Con più varietà sopravvivono molte più creature che con un solo raccolto da 30 passi (70–98%
  contro 44%): un premio frequente e piccolo è più sicuro.
- Per rendere interessante la scelta tra varietà: aumentare il costo dell'azione (rende
  sconveniente il ciclo rapido) o far vedere la varietà prima della maturazione.

### Stagioni (15/11/2026)

Nuovo stimolo `season` (`stimuli/season.py`): un ciclo di stagioni dentro la vita di ogni
creatura. Ogni stagione moltiplica parametri degli altri stimoli, indicati come
`"<stimolo>.<parametro>"`. Parametri stagionali: cibo `regrow_rate` ed `energy_value`;
pericolo `damage`; terreno `growth_rate`, `harvest_energy` e `poison_damage`. La crescita dei
raccolti segue un "orologio di crescita" che avanza con `growth_rate`, quindi un raccolto
piantato in autunno rallenta davvero quando arriva l'inverno. Le creature sentono il momento
dell'anno (`phase_sin`, `phase_cos`; opzionale un canale per stagione); con `random_phase`
ogni vita inizia in un punto a caso dell'anno. Nessuna modifica al core: lo stimolo legge
`agents.energy_gained` e `agents.action` per contare energia, azioni e passi vissuti per
stagione (`stimuli.csv`).

Esperimento `terrain_50_seasons.yaml` (varietà di raccolto come `terrain_50_crops`), anno di
500 passi, quindi due anni per vita:

| stagione | effetti |
|---|---|
| primavera | cibo ricresce ×2, raccolti crescono ×1.5 |
| estate | pericoli ×2 |
| autunno | raccolti rendono ×1.5 |
| inverno | cibo non ricresce, raccolti crescono ×0.2, pericoli ×0.5 |

Seed 42, 150 generazioni (~4 minuti con `--workers 4`). Fitness media: 48 nelle prime 10
generazioni, 2745 nelle ultime 10; vive a fine episodio 75%.

| stagione | energia per passo, gen 0–9 | gen 140–149 | quota di passi con azione, gen 0–9 | gen 140–149 |
|---|---|---|---|---|
| primavera | 0.09 | 4.60 | 12% | 91% |
| estate | 0.11 | 3.44 | 12% | 91% |
| autunno | 0.09 | 4.99 | 3% | 86% |
| inverno | 0.00 | 0.87 | 3% | 84% |

- L'energia segue le stagioni (l'inverno rende 5–6 volte meno), ma il **comportamento quasi
  non cambia**: le creature agiscono in tutte le stagioni, inverno compreso (84%). Coltivare
  sempre resta conveniente perché l'azione costa poco.
- Raccolti per episodio nelle ultime 10 generazioni: slow 37.9, medium 57.4, fast 175.1,
  jackpot 20.2. Il jackpot è raccolto molto più spesso che senza stagioni (fino a 4): da
  verificare con più seed prima di trarne conclusioni.
- Per vedere strategie stagionali (accumulare prima dell'inverno, smettere di seminare quando
  non conviene) servono stagioni più dure: azione più costosa o veleno più forte d'inverno,
  oppure un inverno in cui seminare non rende nulla.

### Vita più cara, soprattutto d'inverno (15/11/2026)

`season` accetta `living_cost` per stagione: un costo extra per ogni passo da vivi, oltre ad
`agent.metabolic_cost`. Anche `terrain.action_cost` ora è stagionale. Due varianti di
`terrain_50_seasons`:

| | `terrain_50_harsh` | `terrain_50_spicy` |
|---|---|---|
| costo base di vivere | 0.15 | 0.08 |
| extra estate / inverno | +0.10 / +0.35 | +0.05 / +0.15 |
| costo dell'azione in inverno | ×4 | ×2 |

Seed 42, 150 generazioni, medie sulle ultime 10:

| | harsh | spicy |
|---|---|---|
| fitness | 89 | 940 |
| vive a fine episodio | 39% | 56% |
| raccolti per episodio | 0 | 144 (di cui 4.9 jackpot) |
| cibo selvatico mangiato | 40 | 1.6 |
| quota di passi con azione: primavera / estate / autunno / inverno | 0.2% in ogni stagione | 88% / 89% / 74% / 72% |

- **Harsh**: troppo dura. All'inizio muoiono tutte; nessuna scopre l'agricoltura, perché
  sperimentare costa troppo. Dopo 150 generazioni sopravvive il 39%, vivendo di cibo selvatico,
  e la sopravvivenza sta ancora crescendo.
- **Spicy**: l'agricoltura emerge comunque, e compare il primo **adattamento stagionale**:
  le creature agiscono meno in autunno e inverno che in primavera ed estate (senza costi
  stagionali la differenza era minima: 84–91%). L'inverno resta in pareggio (0.22 di energia per
  passo).
- Un solo seed: le differenze tra stagioni vanno confermate con più seed.

### Griglie più grandi

La griglia potrà crescere ancora. Misure con 200 agenti e la stessa densità di cibo e pericoli
del 50×50 (1 processo, dopo le correzioni del 15/11/2026):

| griglia | tempo per passo | RAM di picco |
|---|---|---|
| 50×50 | 1.8 ms | 59 MB |
| 100×100 | 3.0 ms | 74 MB |
| 200×200 | 4.0 ms | 134 MB |

Prima delle correzioni il 100×100 usava 2.5 GB e il 200×200 esauriva la memoria (zone di
pericolo costruite con array arene × zone × celle; cibo mangiato cercato in tutte le arene).

Cosa tenere presente salendo ancora:
- la percezione è locale (raggio fisso), quindi il costo per passo cresce solo col numero di
  oggetti (più cibo da far ricrescere), non con l'area;
- la memoria cresce con agenti × celle in modalità `isolated` (griglia del cibo e celle
  visitate per ogni agente): circa 0.45 GB a 500×500, circa 1.8 GB a 1000×1000. Oltre
  conviene ridurre il tipo dei dati delle griglie o passare a strutture sparse;
- su griglie grandi un episodio da 1000 passi copre una parte piccola del mondo: valuta di
  aumentare `world.steps_per_episode` e `sense_radius` insieme alla griglia.

### Conferma con 6 seed (15/11/2026)

Sweep: 5 esperimenti × 6 seed × 300 generazioni (30 run, ~2 ore su 4 core). Report completo,
grafici e test comportamentali in `results/2026-11-15-conferma/` (rigenerabile con
`spiking-evo analyze runs --out report`). Medie sulle ultime 10 generazioni, ± std tra seed.

| esperimento | fitness | vive | cosa si conferma |
|---|---|---|---|
| `food_hazard_50_d3` | 414 ± 67 | 99% | evitamento dei pericoli in tutti i seed: 15 passi su 1000 nei pericoli (1.5%, zone ~7%) |
| `terrain_50_easy` | 2898 ± 917 | 59% | agricoltura in 6 seed su 6; la sopravvivenza non migliora |
| `terrain_50_crops` | 3145 ± 3522 | 82% | **due destini**: 4 seed coltivano, 2 restano raccoglitori di cibo selvatico |
| `terrain_50_seasons` | 3653 ± 1627 | 72% | agricoltura in 6 seed su 6, jackpot raccolto ~42 volte per episodio |
| `terrain_50_spicy` | 1346 ± 1313 | 80% | **due destini**: 3 seed coltivano, 3 raccolgono soltanto (e sopravvivono di più: 89–98% contro 59–76%) |

Cosa **non** si conferma o va ridimensionato:
- l'adattamento stagionale visto col seed 42 (azioni 88% d'estate contro 72% d'inverno) è
  debole sugli altri seed: tra 0 e 9 punti percentuali di differenza. Nel test comportamentale
  dell'agricoltore migliore (spicy seed 3) rendere la creatura cieca alla stagione non cambia
  nulla: il senso della stagione non viene usato;
- i risultati di un singolo seed su terreno e stagioni non sono affidabili: tra seed la fitness
  varia di un ordine di grandezza.

Test comportamentali (30 episodi nuovi):
- evitatore di pericoli (hazard seed 1): cieco ai pericoli passa 5.5 volte più tempo nelle zone
  (63.6 passi contro 11.6); cieco al cibo raccoglie un quinto dell'energia;
- agricoltore (spicy seed 3): cieco al terreno sopravvive il 23% invece del 70% e dimezza i
  raccolti; cieco ai pericoli ci passa 15 volte più tempo; cieco alle stagioni: nessun effetto.

Prossimo passo proposto: continuità tra ambienti (popolazione che porta con sé ciò che ha
imparato) e plasticità sinaptica durante la vita.

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

1. ~~`stimuli/hazard.py` → `HazardStimulus`~~ **fatto** (vedi sezione 2). La percezione
   egocentrica è ora la funzione condivisa `egocentric_sense` in `food.py`.
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
   `world.arenas: shared`, quindi niente parallelismo a blocchi. Per "emettere" può usare
   un'uscita d'azione generica (`network.action_outputs`), già disponibile.

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
