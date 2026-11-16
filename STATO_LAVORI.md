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
| Test | 150, tutti verdi anche su Windows 11 / Python 3.12 (16/11/2026); `ruff`: 3 righe troppo lunghe in `scripts/compare_d4.py` e `compare_d5.py` |
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
sensori. La ricerca mirata del cibo è dimostrata dai test della sezione successiva.

### Prestazioni (misurate su un container con 4 core)

| modalità | tempo per generazione |
|---|---|
| prima implementazione | 9.9 s |
| propagazione sparsa degli spike | 1.6 s |
| ottimizzazioni attuali, 1 processo | 1.3 s |
| 1 run con `--workers 4` | 0.84 s |
| `sweep` di 4 seed in parallelo | 0.47 s per run |

Un baseline da 300 generazioni dura circa 6.5 minuti su un core.

### Baseline con 8 seed e strategia evoluta (14/11/2026, PC Windows)

`sweep food_only_baseline.yaml --seeds 1-8`, 300 generazioni, circa 22 minuti su 16 core
logici (circa 6 s per generazione con 8 run in parallelo).

| generazione | 0 | 49 | 99 | 199 | 299 |
|---|---|---|---|---|---|
| fitness media (media tra seed) | 16 | 444 | 514 | 550 | 548 |
| deviazione standard tra seed | 4 | 90 | 26 | 15 | 15 |

Plateau intorno a 550 dalla generazione 200 circa: per questa config bastano 200
generazioni. I seed partiti peggio recuperano e la varianza tra seed si stringe. A parità di
fitness le reti finali hanno da 15 a 48 sinapsi e firing rate da circa 20 a 80 Hz.

**Cosa hanno imparato.** `scripts/behaviour_baseline.py` valuta il genoma migliore di gen 299
di ogni seed su 200 episodi nuovi, manipolando i sensori prima della rete:

| condizione | energia raccolta (media sugli 8 seed) |
|---|---|
| sensori intatti | 561 (541–571) |
| sensori del cibo azzerati | 9: vanno dritti, nel mondo toroidale ripercorrono la stessa riga di 40 celle e muoiono di fame |
| solo `here` (front/left/right azzerati) | 9, identico al precedente |
| sinistra e destra scambiate | 119 |
| camminatore casuale | 51 |
| regola scritta a mano "gira verso il lato col segnale più forte" | 769 |

La strategia è la stessa su tutti i seed: un **veicolo di Braitenberg**. Senza cibo vanno
dritti (girano in meno dell'1,5% dei passi); con cibo su un lato girano verso di esso in circa
il 56% dei passi e quasi mai dall'altra parte. Ne sono emerse due varianti speculari che
rendono uguale:

- seed 1, 6, 7 camminano **in avanti** con collegamenti ipsilaterali (`left -> gira_sx`);
- seed 2, 3, 4, 5, 8 camminano **all'indietro** con collegamenti controlaterali
  (`left -> gira_dx`), che in retromarcia è la sterzata giusta.

Nella rete del seed 1 (15 sinapsi) il circuito si legge direttamente: `bias -> avanti`,
`left -> gira_sx`, `right -> gira_dx`.

Conseguenze:

- il canale `here` è quasi sempre 0 quando viene letto, perché il cibo è mangiato appena
  l'agente arriva sulla cella: azzerarlo non cambia nulla;
- `front` è poco usato, e chi cammina all'indietro non vede dove va (manca `back`);
- senza cibo in vista non c'è alcuna strategia di ricerca: vanno dritti;
- la regola scritta a mano rende il 37% in più, quindi c'è margine di miglioramento.

Il comando `behaviour` fa i test di cecità per stimolo e torna con questi numeri (seed 1:
566 normale, 8 cieco). Lo script aggiunge i test che `behaviour` non fa: scambio
sinistra/destra, ablazione dei soli canali direzionali, confronto con un camminatore casuale
e con una regola di riferimento, e statistiche di sterzata. Rieseguito sul codice attuale
(16/11/2026) dà risultati identici, numero per numero.

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

### Continuità tra ambienti e plasticità: confronto su RunPod (15/11/2026)

**Continuità (tappa A).** Fasi d'ambiente dentro un run (`phases`, cumulative) e partenza da
un'altra popolazione (`--init-from`). Quando compaiono sensi o uscite nuovi, i neuroni nuovi
vengono collegati a caso come in un cervello appena nato (`adapt_genome(..., rng)`). Una prima
versione li lasciava scollegati: il neurone "agisci" non scattava mai e nessuna popolazione
diventava agricoltrice (risultato scartato, archiviato fuori dalle analisi).

**Plasticità (tappa B).** STDP modulata dalla ricompensa: tracce di idoneità per sinapsi, la
variazione di energia rispetto alla media recente fa da ricompensa. Il genoma porta una
velocità di apprendimento e quali sinapsi sono plastiche; entrambe evolvono. Pesi riportati a
quelli del genoma a ogni nascita. Costo: circa +40% di tempo.

**Esperimento** (`runpod/curriculum_job.sh`, pod cpu3c da 32 vCPU, 27 minuti di calcolo, ~0.67 $):
percorso `curriculum_50` (cibo → pericoli → agricoltura → stagioni pepate → shock con inverno
durissimo e metà del cibo) senza e con plasticità, contro popolazioni da zero nell'ambiente
finale (`terrain_50_spicy`, poi `shock_50`); 8 seed ciascuno. Dettagli in
`results/2026-11-15-curriculum-runpod/`.

| confronto | da zero | percorso, cervello fisso | percorso, cervello plastico |
|---|---|---|---|
| arrivo nell'ambiente pepato (prime 10 gen): fitness | 52 | 1517 | 1943 |
| arrivo: vive a fine episodio | 16% | 77% | 73% |
| dopo 150 gen nell'ambiente pepato: fitness | 457 | 2785 | 2668 |
| dopo 150 gen: seed agricoltori | 3/8 | 5/8 | 3/8 |
| (da zero dopo 300 gen: fitness 1259, agricoltori 4/8) | | | |
| shock, prime 5 gen: fitness | 582 | 1356 | 1951 |
| shock, ultime 10 gen: fitness / vive | 899 / 64% | 1762 / 63% | 1857 / 54% |

Cosa dicono i dati:
- **La continuità aiuta molto.** Chi arriva con una storia sopravvive da subito (77% contro
  16%) e in 150 generazioni fa 6 volte meglio di chi parte da zero; anche dopo il doppio delle
  generazioni chi parte da zero resta indietro. Nello shock le popolazioni con storia rendono
  il doppio, ma non sopravvivono di più.
- **La plasticità, così com'è, non aiuta.** Le differenze tra cervello fisso e plastico sono
  piccole rispetto alla variabilità tra seed. E l'evoluzione **la spegne**: la velocità di
  apprendimento media scende da 0.05 a 0.014 in 100 generazioni e a 0.006–0.008 alla fine.
  Imparare durante la vita, con questa regola e questa ricompensa, costa più di quanto rende.
- Possibili motivi, da verificare: la ricompensa (variazione di energia) è rumorosa e include
  costi costanti; l'ambiente cambia soprattutto *tra* generazioni, poco *dentro* una vita (il
  senso della stagione non viene usato); la regola STDP è semplice.

Prossimi passi possibili: ambienti che cambiano dentro la vita in modo imprevedibile (es.
campi e pericoli che si spostano a metà vita), una ricompensa più pulita, e la tappa C
(cervello che nasce piccolo e cresce), che dà alla plasticità neuroni nuovi su cui lavorare.

### Plasticità in un mondo che cambia durante la vita (15/11/2026)

Nuove opzioni: `relocate_every` per `hazard` e `terrain` (zone e suolo nascosto si spostano
durante l'episodio; i raccolti su suolo cambiato muoiono) e `plasticity.reward: stimuli`
(ricompensa = solo energia data o tolta dagli stimoli, senza il costo fisso di vivere).

Esperimento su RunPod (pod cpu3c da 32 vCPU, 14 minuti di calcolo, ~0.53 $ in tutto):
4 condizioni × 8 seed × 300 generazioni, ambiente come `terrain_50_crops`; nel mondo che
cambia pericoli e campi si spostano ogni 250 passi (4 mappe per vita). Dettagli in
`results/2026-11-15-dynamic-runpod/`. Medie sulle ultime 10 generazioni:

| | cervello fisso | cervello plastico |
|---|---|---|
| mondo fermo: fitness | **2876 ± 3130** | 1892 ± 1920 |
| mondo fermo: agricoltori | 6/8 | 6/8 |
| mondo che cambia: fitness | **1320 ± 1485** | 897 ± 743 |
| mondo che cambia: agricoltori | 5/8 | 3/8 |

Velocità di apprendimento media (plastici): 0.05 iniziale, minimo ~0.017–0.018 attorno alla
generazione 50–100, poi risale a 0.028–0.030; quasi uguale nei due mondi.

Conclusione: **l'ipotesi non è confermata.** Con questa regola (STDP modulata dalla
ricompensa) la plasticità peggiora i risultati in entrambi i mondi, e di più in quello che
cambia; l'evoluzione non la spegne del tutto, ma non la favorisce nel mondo variabile.
Probabile causa: la regola modifica le sinapsi in base a coincidenze di spike e a una
ricompensa che arriva in ritardo (raccolti dopo decine di passi, spostamenti ogni 250): il
segnale è troppo debole e rumoroso per imparare *dove* andare, e intanto disturba i
comportamenti che l'evoluzione aveva già fissato. Strade possibili: regole più semplici e
mirate (es. apprendimento associativo solo sulle sinapsi sensori→motori, o "abituazione"),
segnali di ricompensa immediati, oppure plasticità limitata ai neuroni nuovi (tappa C).

### Tappa C: cervello che nasce piccolo e cresce (15/11/2026)

`network.development`: i neuroni nascosti possono essere **dormienti alla nascita** (gene
`dormant`) e svegliarsi durante la vita quando la "difficoltà" accumulata (energia tolta dagli
stimoli: danni, veleno, costi delle azioni) supera la loro soglia (gene `wake_at`). Ogni
neurone sveglio costa `neuron_cost` energia per passo. Con `plastic_new_only` la plasticità
agisce solo sulle sinapsi dei neuroni risvegliati. Unica aggiunta al core: un controller può
dichiarare `energy_cost()`, pagato insieme al costo metabolico.

Esperimento su RunPod (pod cpu3c 32 vCPU, ~37 minuti, ~0.62 $): mondo che cambia (zone spostate
ogni 250 passi), 10 neuroni nascosti iniziali, costo 0.002 per neurone sveglio, 8 seed ×
300 generazioni; confronto con i run precedenti `dynamic_fixed_50` e `dynamic_plastic_50`.
Dettagli in `results/2026-11-15-develop-runpod/`. Ultime 10 generazioni:

| condizione | fitness | vive | agricoltori | neuroni dormienti alla nascita → svegliati |
|---|---|---|---|---|
| cervello fisso (test precedente, 5 neuroni, nessun costo) | 1320 ± 1485 | 86% | 5/8 | – |
| cervello plastico (test precedente) | 897 ± 743 | 86% | 3/8 | – |
| tutti svegli dalla nascita, con costo (controllo) | 1795 ± 1187 | 84% | 6/8 | 0 → 0 |
| **nasce piccolo e cresce** | **2423 ± 1139** | 78% | **8/8** | 5.8 → 5.3 |
| crescita + apprendimento solo sui neuroni nuovi | 1093 ± 1027 | 83% | 3/8 | 4.3 → 2.5 |
| crescita + apprendimento su tutto | 1473 ± 1178 | 82% | 5/8 | 5.4 → 4.1 |

- *Smentito dalla conferma con 24 seed (sezione successiva):* con 8 seed lo sviluppo
  sembrava migliore del controllo (+35%, agricoltura in 8 seed su 8); con 24 seed la
  differenza sparisce.
- Quasi tutti i neuroni dormienti si svegliano entro la vita (5.3 su 5.8): l'evoluzione usa
  lo sviluppo come **calendario di crescita** guidato dalle difficoltà, non come riserva
  lasciata spenta. Il vantaggio può venire dal risparmio di energia nei primi passi, o dal
  fatto che il cervello si "completa" quando la creatura incontra i problemi.
- **La plasticità continua a peggiorare** i risultati, anche limitata ai neuroni nuovi.
  Curiosità: quando l'apprendimento è limitato ai neuroni nuovi l'evoluzione **alza** la
  velocità di apprendimento (0.05 → 0.064), mentre quando agisce su tutto la abbassa a
  0.007: imparare sui neuroni nuovi è tollerato, toccare quelli vecchi è dannoso.

### Tappa C, conferma con 24 seed: il vantaggio NON regge (15/11/2026)

Altri 16 seed (9–24) per `dynamic_develop_50` e `dynamic_awake_cost_50`, uniti agli 8 già
fatti (pod cpu3c 32 vCPU, ~0.73 $ compreso un primo pod bloccato e terminato). Ultime 10
generazioni, 24 seed per condizione (`results/2026-11-15-develop-confirm/`):

| condizione | fitness (media ± std) | mediana | agricoltori | vive |
|---|---|---|---|---|
| tutti svegli, con costo | 1646 ± 1184 | 1593 | 16/24 | 82% |
| nasce piccolo e cresce | 1580 ± 1227 | 1273 | 17/24 | 81% |

Differenza −4%, t di Welch −0.19, p (permutazione, una coda) 0.58: **nessuna differenza**.
Nei primi 8 seed lo sviluppo era avanti (2423 contro 1795), nei 16 nuovi è indietro
(1159 contro 1572): il vantaggio visto prima era **dovuto al caso**. La variabilità tra seed
(±1200) è enorme rispetto agli effetti che cerchiamo: per esperimenti di questo tipo servono
almeno 24–32 seed per condizione, e conviene dichiarare prima quale test si userà.

Lo sviluppo in sé funziona come previsto (circa 5 neuroni dormienti alla nascita, 4 svegliati
durante la vita) ma, con questi parametri, non dà né un vantaggio né uno svantaggio misurabile.

### Tappa D1: ecosistema con nascite e morti (15/11/2026)

Nuovo modo di simulare, `spiking-evo ecosystem CONFIG` (`spiking_evo/ecosystem/sim.py`): un
solo mondo condiviso, tempo continuo, nessun punteggio. Le creature nascono, crescono,
si riproducono, invecchiano e muoiono; si diffonde ciò che lascia discendenti vivi.

- **Riproduzione**: un adulto con energia sopra la soglia (gene) genera `clutch` figli (gene)
  accanto a sé, cedendo una quota della sua energia (`endowment`, gene; il 10% va perso).
  Figli = genoma del genitore con mutazioni, geni del ciclo di vita mutati.
- **Piccoli fragili** per `juvenile_age` passi (gene): metà velocità, doppio danno dagli
  stimoli, niente azioni, niente riproduzione.
- **Morte** per fame o per età massima (3000 passi).
- Nuovo stimolo `body`: la creatura sente la propria energia e se è piccola.
- Modifiche al core (generiche): `age` e `stage` negli agenti, `World.spawn` per far nascere
  creature in posti liberi; il controller sostituisce le reti senza azzerare lo stato dei vivi.
- Con cervelli casuali la popolazione si estingue in ~3000 passi (non trovano cibo): si parte
  da una specie esistente, `ecosystem.init_from` →
  `experiments/seed_populations/foragers.json.gz` (raccoglitori evoluti in
  `food_hazard_50_d3`, seed 3, 300 generazioni).

Prima prova (`eco_d1.yaml`, seed 1, 50 000 passi, ~3.5 minuti in locale):

| passo | popolazione | generazioni | discendenze | figli per parto | dote | energia per riprodursi | infanzia (passi) |
|---|---|---|---|---|---|---|---|
| 250 | 121 | 1 | 115 | 2.00 | 0.50 | 100 | 151 |
| 10 000 | 85 | 9 | 11 | 2.16 | 0.56 | 85 | 158 |
| 20 000 | 92 | 16 | 4 | 1.92 | 0.66 | 60 | 199 |
| 30 000 | 107 | 31 | 1 | 1.64 | 0.71 | 48 | 197 |
| 50 000 | 105 | 50 | 1 | 1.69 | 0.47 | 41 | 113 |

- La popolazione è stabile (min 35, media 105, max 136 su 300 posti); intorno al passo 3000
  la generazione iniziale muore quasi tutta di vecchiaia insieme e i discendenti la
  rimpiazzano.
- Una sola discendenza (dei 120 fondatori) conquista tutto entro il passo 30 000.
- I geni del ciclo di vita evolvono: le creature si riproducono prima (soglia da 100 a 41
  energia) e per un lungo tratto fanno **meno figli con più dote** (1.6 figli, 71% della
  propria energia) e un'infanzia più lunga; alla fine dote e infanzia tornano a scendere.
- Un solo seed: tendenze da confermare. Prossimo passo D2: crescita del cervello che costa
  energia, senso della parentela, azione "nutri".

### Tappa D2: crescita che costa, parentela, nutrimento (15/11/2026)

- **Crescita che costa energia** (`network.development.trigger: growth`): i neuroni dormienti
  si svegliano quando la creatura ha investito abbastanza energia nel cervello
  (`ecosystem.growth_rate` per passo, sopra una riserva `growth_reserve`; soglia `wake_at`
  per neurone, gene). Nei fondatori presi da una specie senza sviluppo, metà dei neuroni
  nascosti viene resa dormiente.
- **Stimolo `kin`** (`stimuli/kin.py`): senso direzionale dei propri piccoli (`child_*`, con
  `child_here` = uno è a portata) e del proprio genitore (`parent_*`); azione "nutri": un
  adulto cede `feed_amount` energia al proprio piccolo più povero a portata, che ne riceve
  `feed_efficiency`. Con `kin_only: false` si può nutrire qualunque piccolo. Nessun obbligo:
  lo decide il cervello evoluto.
- Il core espone `world.extra` (ids e genitori pubblicati dall'ecosistema); calcolo dei
  rapporti solo tra creature vive (3.9 ms per passo con 300 posti).
- Esperimento `eco_d2.yaml`. Prima prova (seed 1, 10 000 passi): la popolazione regge
  (67–117), i piccoli hanno meno neuroni svegli degli adulti (7–8 contro 12) e crescono;
  il nutrimento avviene di rado (1–14 volte ogni 250 passi), cioè per ora per caso.

### Tappa D3: l'accudimento NON emerge (15/11/2026)

Esperimento su RunPod (pod cpu3c 32 vCPU, ~30 minuti, ~0.48 $): 4 condizioni × 24 seed ×
50 000 passi, ecosistema D2; mondo abbondante (`eco_d3_mild_*`) o duro (meno cibo, piccoli
3 volte più fragili, `eco_d3_harsh_*`); nutrimento attivo (`*_care`) o controllo con stessi
sensi e stessa azione ma senza trasferimento (`*_control`). Ipotesi e test fissati prima
(`scripts/compare_eco.py`); risultati in `results/2026-11-15-eco-d3/`.

| condizione | popolazione | nutrimenti ogni 250 passi | figli per parto | dote | infanzia |
|---|---|---|---|---|---|
| abbondante, nutrimento | 78.9 | 10.2 | 2.47 | 0.51 | 95 |
| abbondante, controllo | 77.6 | 7.6 | 2.28 | 0.58 | 103 |
| duro, nutrimento | 32.8 | 3.1 | 2.29 | 0.56 | 125 |
| duro, controllo | 33.2 | 5.6 | 2.18 | 0.55 | 97 |

Test pre-registrati (permutazione, una coda, alpha 0.05):
- H1 (il nutrimento viene selezionato): **no** in entrambi i mondi (p 0.31 e 0.92).
- H2 (il nutrimento conviene): popolazione leggermente più alta nel mondo abbondante (+1.8,
  p 0.035), ma mortalità dei piccoli **non** più bassa (p 0.96); nel mondo duro nessun effetto.
- H3 (il mondo duro favorisce l'accudimento): **no** (p 0.92).
- Nessuna estinzione in 96 popolazioni.

Lettura: il nutrimento resta un evento raro e casuale (circa 0.5 per 1000 passi di adulto,
qualche unità di energia contro centinaia raccolte dal cibo) e l'evoluzione non lo spinge.
Probabili ragioni: il beneficio per il piccolo è piccolo (crescita da ~15 di energia,
infanzia di ~100 passi), il genitore deve trovarsi adiacente al figlio proprio mentre il
neurone "nutri" scatta, e la riproduzione asessuata con figli vicini rende la parentela
quasi uniforme. Come in natura, senza un forte bisogno del piccolo le cure non convengono.
Cosa cambiare per dare all'accudimento una possibilità: piccoli che **non sanno nutrirsi
da soli** per un tempo lungo (es. non possono mangiare il cibo selvatico), dote alla
nascita bassa, figli che seguono il genitore (il senso `parent_*` esiste già).

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
12. **Arene a gruppi (modifica al core, tappa E0, 16/11/2026):** `world.arenas: groups` con
    `world.group_size`. A ogni episodio gli agenti vengono divisi a caso (da
    `(seed, chiavi, "groups")`) in mondi da `group_size` agenti che condividono le risorse e
    possono percepirsi; la fitness resta individuale. Non si può dividere tra processi (come
    `shared`): il parallelismo si fa tra seed. `World.per_arena` ora estrae una riga per arena
    quando la popolazione non è divisa; per `isolated` e `shared` è identico a prima
    (verificato bit per bit sulla baseline).
13. **Gruppi di cloni (tappa E1, 16/11/2026):** `genetics.clones: k` valuta ogni genoma come
    k agenti identici che vivono nella stessa arena di gruppo; la fitness del genoma è la
    media del gruppo (selezione tra gruppi di parenti, come nelle colonie di robot di
    Floreano e colleghi). Richiede `world.arenas: groups`, `world.group_size == k` e
    `world.group_shuffle: false` (nuova opzione: gli agenti consecutivi restano insieme
    invece di essere rimescolati). `genetics.population` conta i genomi, quindi gli agenti
    per generazione sono `population * k`. Implementato in `Evolution.evaluate` (espande,
    valuta, fa la media con `collapse_clones`), senza toccare il core oltre a
    `group_shuffle`.

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

### Windows

Verificato il 16/11/2026 su Windows 11 con Python 3.12: i 150 test passano, `sweep`
parallelo (processi `spawn`), `analyze` e `behaviour` funzionano. `runpod/*.sh` e il
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


## D4 naturale (2026-11-15) — cure NON emergono, vengono eliminate

Vedi `results/2026-11-15-eco-d4/README.md`. Con cure: 8/32 estinti, feed rate finale 13 contro 109 del controllo (H1 smentito, p=1.0); H2 e H3 non supportati. Nutrire costa e, senza parentela (kin_only false) e con popolazione sempre al tetto di 300, non rende: la selezione lo spegne. Prossimi passi possibili: kin_only true (dono solo ai propri figli, selezione di parentela), capacità più alta o mondo limitato dal cibo e non dai posti, piccoli che senza aiuto davvero non sopravvivono per scarsa capacità (cervello piccolo) e non per regola.

## D5 (2026-11-16) — mondo autoregolato, cure né eliminate né selezionate

Vedi `results/2026-11-16-eco-d5/README.md`. Lezioni di calibrazione (test brevi locali): i
fondatori "foragers" si avvelenavano (uscita azione nuova cablata a caso) → fondatori evoluti
300 generazioni nel mondo D5; con terreno senza limite la popolazione esplodeva → riposo del
terreno (`fallow_time`); il cibo condiviso va scalato per la popolazione (500 → ~80 individui,
7 ms/passo). Esito 16+16: H1–H3 non supportati (p ≈ 0.47–0.62), nessuna estinzione. Il gesto
"nutri" deriva allo stesso modo con e senza effetto. Prossimo passo proposto: rendere il dono
energeticamente rilevante (feed_amount più alto) e capire perché la mortalità giovanile è 70–90%.

## D6 (2026-11-16) — dono da 10: le cure ancora non emergono

Vedi `results/2026-11-16-eco-d6/README.md`. D5 con `feed_amount: 10` (8 ricevuti), 16+16 mondi
su RunPod (circa 65 min, circa 1.05 $), nessuna estinzione. H1–H3 non supportati (p = 0.92,
0.92, 0.85): con il dono grande si nutre semmai meno (2.7 contro 6.8 gesti ogni 1000
passi-adulto, non significativo, p = 0.08 nella direzione opposta) e la mortalità giovanile
resta circa 0.9 in quasi tutti i mondi. Il controllo riproduce esattamente i numeri del
controllo D5 (stessa config e seed): simulazione deterministica. Il notebook (Ryzen 7 5700U,
8 core fisici) non basta per questi esperimenti: 16 mondi in parallelo vanno a circa 75 s ogni
1000 passi, quindi circa 3.5 ore per 8+8, contro circa 14 s sul pod da 32 vCPU.

## E0 (2026-11-16) — vedere gli altri: preparazione e ipotesi

Primo passo verso la comunicazione (percorso E0 → E1 segnale → E2 più "parole" → E3
coordinazione). Nuovi pezzi: arene a gruppi (decisione 12), stimolo `others` (dove sono gli
altri agenti vivi del gruppo, per settore, raggio configurabile; `sense_gain: 0` = controllo
con gli stessi ingressi sempre a 0), stimolo `food_patch` (cibo in chiazze gaussiane che si
spostano ogni `relocate_every` passi; il cibo ricresce attorno ai centri attuali).

Esperimento: `e0_social` contro `e0_control`, gruppi da 10, 40×40, 60 cibi in 3 chiazze
(raggio 1.5, si spostano ogni 250 passi, cibo sentito solo entro 2 celle), gli altri sentiti
entro 6 celle; 200 generazioni, 24 seed per condizione. Ipotesi fissate prima del lancio in
`scripts/compare_e0.py`: H1 più energia con il senso sociale; H2 nella condizione sociale la
popolazione finale raccoglie meno se resa sorda agli altri. Pilota (1 seed, 30 generazioni,
solo calibrazione): il mondo funziona (fitness da 15 a circa 200–240), circa 2.8 s per
generazione con `others`, 1.7 s senza.

### E0 — risultati (2026-11-16)

Vedi `results/2026-11-16-e0/README.md`. 24+24 seed su RunPod (circa 23 min, circa 0.37 $).
**H2 supportata** (p = 0.010): la popolazione sociale finale raccoglie meno se resa sorda agli
altri (+2.4 di energia in media, 17 seed su 24). **H1 non supportata** (+2.4% di energia,
p = 0.22). Esplorativo: il vantaggio sociale è netto nelle prime 50 generazioni (+15 di
energia) e si riduce dopo; la sopravvivenza resta più alta per tutta la run (98% contro 94–95%).
L'informazione sui vicini è un indizio involontario che porta con sé competizione; prossimo
passo E1: un segnale emesso per scelta (`signal`), con gruppi di cloni contro non imparentati.

## E1 (2026-11-16) — il primo segnale: preparazione e ipotesi

Nuovo stimolo `signal` (`spiking_evo/stimuli/signal.py`): un'uscita d'azione generica; chi
la attiva emette per quel passo (costo 0.01) e i membri vivi del gruppo lo sentono al passo
dopo, per settore, entro 6 celle; nessuno sente sé stesso; `sense_gain: 0` = canale muto. Lo
stimolo conta le emissioni separando i passi "nutriti" (energia guadagnata negli ultimi 10
passi) da quelli "a digiuno": serve per misurare se il segnale porta informazione.

Disegno 2×2, mondo di E0 con `others` acceso ovunque, 400 agenti per generazione, 300
generazioni, 24 seed per condizione:

| | segnale sentito | canale muto |
|---|---|---|
| gruppi di 10 cloni, fitness di gruppo (40 genomi) | `e1_clones` | `e1_clones_mute` |
| individui non imparentati, gruppi rimescolati (400 genomi) | `e1_mixed` | `e1_mixed_mute` |

Ipotesi fissate prima del lancio in `scripts/compare_e1.py`, misurate sulla popolazione
finale su 10 episodi nuovi: H1 informatività (P(emette | nutrito) − P(emette | a digiuno))
più alta in `e1_clones` che in `e1_clones_mute`; H2 in `e1_clones` la popolazione raccoglie
meno se resa sorda al segnale; H3 informatività più alta in `e1_clones` che in `e1_mixed`.
Atteso dalla letteratura: tra individui non imparentati e con selezione individuale emettere
costa a chi emette e aiuta gli altri, quindi dovrebbe essere selezionato contro.

**Pilota E1 sul notebook (16/11, 1 seed, 20 generazioni, solo calibrazione).** Tutte e 4 le
condizioni girano; `compare_e1.py` funziona (controllo di coerenza: costo della sordità = 0
nei muti). Tempi sul notebook circa 15 s per generazione con 4 run in parallelo. **Problema di
calibrazione:** nei gruppi di cloni l'emissione sale al massimo entro la generazione 5 (circa
980 emissioni su 1000 passi), sia con segnale sentito sia con canale muto, insieme
all'attività delle reti (da 31 a circa 100 Hz); tra non imparentati sale fino a circa 300
alla generazione 5 e poi scende a circa 90 (il costo la seleziona contro). Con
`emit_cost: 0.05` (circa 47 di energia per episodio se si emette sempre) i cloni emettono
ancora circa 940 volte su 1000. Un segnale sempre acceso non porta informazione: con questo
disegno H1 può fallire per un effetto soglia. Da decidere con l'utente prima del lancio
(le config restano con `emit_cost: 0.01`, come pre-registrato).
