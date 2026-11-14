# Spiking-Evo

Popolazioni di agenti guidati da reti di neuroni spiking (LIF) evolute con un algoritmo
genetico, in un mondo a griglia estendibile tramite stimoli. Stato: **milestone 1
(baseline)** completo. Specifica completa in `SPEC.md`; stato dei lavori e prossimi passi in
`STATO_LAVORI.md`.

## Uso

```bash
pip install -e ".[dev]"
python -m spiking_evo run spiking_evo/experiments/food_only_baseline.yaml
python -m spiking_evo run CONFIG --seed 7 --generations 100 --out runs   # override
python -m spiking_evo run CONFIG --resume                                # riprende da checkpoint.json
python -m spiking_evo run CONFIG --workers 4                             # valutazione su 4 processi
python -m spiking_evo sweep CONFIG [CONFIG ...] --seeds 1-16             # più run in parallelo
python -m spiking_evo evaluate runs/food_only_baseline/42/snapshots/gen_0099.json \
    --config spiking_evo/experiments/food_only_baseline.yaml --episodes 20
pytest && ruff check .
```

Output in `runs/<esperimento>/<seed>/`: `generations.csv`, `terms.csv`,
`snapshots/gen_XXXX.json`, `checkpoint.json`, `config.yaml` (con commit hash e flag dirty).

## Risultato baseline (seed 42, 100 generazioni, ~3 min su CPU)

| | gen 0 | gen 99 |
|---|---|---|
| fitness media | 19.9 | 515.8 |
| agenti vivi a fine episodio | 73.5% | 100% |
| celle distinte visitate (media) | 81 | 689 |

Media delle prime 10 generazioni 193, delle ultime 10 519. Il genoma migliore, rivalutato su
20 episodi nuovi, raccoglie 540 di energia; con i sensori del cibo azzerati ne raccoglie 8,
quindi il comportamento evoluto dipende dagli input sensoriali.

## Scelte di progetto e scostamenti dalla specifica

- **Interfaccia `Stimulus` vettorizzata.** `sense` e `interact` ricevono l'intero batch di
  agenti (struttura di array) e restituiscono una riga per agente, invece di una chiamata per
  agente. Una chiamata Python per agente costerebbe ~10^6 chiamate a generazione; la semantica
  è la stessa. Vedi `stimuli/base.py`.
- **Arene.** Con `world.arenas: isolated` (default) ogni agente vive in una copia privata del
  mondo: la fitness non dipende dagli altri individui. `shared` mette tutti nello stesso mondo
  (competizione per il cibo), già supportato da `FoodStimulus`.
- **Simulazione batch.** Tutta la popolazione è simulata in parallelo, con propagazione
  sparsa degli spike (gather + bincount sulle sinapsi esistenti) e un ring buffer per i
  ritardi: ~1.3 s a generazione su un core per 200 agenti × 1000 passi × 10 passi SNN.
- **Parallelismo riproducibile.** Con `--workers N` la popolazione è divisa in blocchi
  valutati da N processi. Ogni estrazione casuale per agente o per arena avviene sull'intera
  popolazione e poi viene affettata (`World.per_arena`), quindi i risultati sono identici
  per qualunque numero di worker (testato). Le statistiche di attività contano solo la vita
  dell'agente.
- **LIF.** Integrazione esatta dell'esponenziale con salti sinaptici istantanei; durante la
  refrattarietà gli input vengono ignorati. Parametri per neurone (soglia e `tau_m` evolvibili).
- **Ritardi.** Interi in `[1, max_delay]` passi SNN; uno spike emesso a `t` arriva a `t + d`.
- **Sensori del cibo.** Egocentrici: settori fronte/sinistra/destra/dietro (coni di 90°) entro
  `sense_radius`, pesati 1/distanza e compressi in [0, 1], più il canale `here`. I canali usati
  si scelgono nella config dello stimolo; la dimensione dell'input si calcola da sola. Un canale
  `bias` costante è opzionale (`network.bias_input`).
- **Encoder.** `rate` (Bernoulli), `latency` (time-to-first-spike nella finestra del tick),
  `current` (corrente costante nei neuroni di input).
- **Motori.** Quattro neuroni d'uscita in due coppie antagoniste (avanti/indietro,
  sinistra/destra), decodificate come differenza di rate sulla finestra del tick.
- **Mutazioni.** I tassi per gene (`weight`, `threshold`, `tau`, `delay`) sono probabilità per
  sinapsi/neurone; quelli strutturali (`add_synapse`, `remove_synapse`, `add_neuron`,
  `remove_neuron`) sono probabilità per genoma. `add_neuron` spezza una sinapsi (stile NEAT).
- **Crossover** (opzionale, `genetics.crossover`, default 0): uniforme, allineato per indice di
  neurone e per coppia (src, dst); le sinapsi presenti solo nel genitore peggiore non passano.
- **Fitness.** `energy` = energia ottenuta dagli stimoli durante la vita; `firing_penalty` =
  `-w * max(0, rate - target) / target`; `silence_penalty` = `-w` se nessun neurone non di input
  spara mai; `exploration` = `w * celle distinte`. Ogni termine è loggato in `terms.csv`.
- **Riproducibilità.** Ogni componente stocastico ha un generatore derivato dal seed globale con
  chiavi esplicite (`seeding.py`): popolazione iniziale, episodio di valutazione per generazione,
  riproduzione per generazione. Un run ripreso da checkpoint produce gli stessi genomi di un run
  ininterrotto (testato).
- **Dipendenze.** Solo NumPy e PyYAML a runtime (PyYAML per le config YAML; TOML via `tomllib`
  della stdlib). Nessun backend Brian2/Norse per ora.

## Esecuzione su Runpod

La simulazione è CPU-bound e non usa GPU: le reti sono piccole e sparse, e ogni generazione
è una sequenza di ~10.000 passi SNN in cui l'overhead per passo, non il calcolo, domina. Un
pod GPU costerebbe di più senza andare più veloce: conviene un **pod CPU** con molti vCPU
(meglio la classe *compute-optimized*) e un **network volume** su `/workspace` per i risultati.

Come usare i core (misurato su 4 core, baseline 200 agenti):

| modalità | tempo per generazione-run |
|---|---|
| 1 run, 1 processo | 1.3 s |
| 1 run, `--workers 4` | 0.84 s |
| `sweep` di 4 seed, 1 processo ciascuno | 0.47 s |

Per esperimenti di ricerca (più seed) usa `sweep`, che scala quasi linearmente coi core;
`--workers` serve quando conta il tempo di un singolo run. `sweep` usa per default tanti
job quanti sono i vCPU realmente assegnati al container (quota cgroup: `nproc` sui pod può
mostrare i core dell'host; verifica con `python -m spiking_evo cpus`).

Robusto alle interruzioni (pod interrompibili/spot inclusi): `sweep` lancia ogni run con
`--resume` e checkpoint a ogni generazione; rilanciando lo stesso comando i run finiti vengono
saltati e gli altri ripartono dall'ultima generazione completata, con risultati identici a un
run ininterrotto. Ogni run scrive `run.log` nella propria cartella; un riepilogo finisce in
`<out>/sweep_<timestamp>.csv`.

### Opzione A: immagine Docker (consigliata)

```bash
docker build --platform=linux/amd64 --build-arg GIT_COMMIT=$(git rev-parse HEAD) \
    -t <registry>/spiking-evo:0.1.0 spiking-evo/
docker push <registry>/spiking-evo:0.1.0
```

Template del pod: immagine `<registry>/spiking-evo:0.1.0`, network volume su `/workspace`,
porta TCP 22 per SSH, e queste variabili d'ambiente:

| variabile | esempio |
|---|---|
| `SWEEP_CONFIGS` | `/opt/spiking-evo/spiking_evo/experiments/food_only_baseline.yaml` |
| `SWEEP_SEEDS` | `1-32` |
| `SWEEP_ARGS` | `--generations 300` (opzionale) |

All'avvio `runpod/start.sh` configura SSH con `$PUBLIC_KEY`, lancia lo sweep (output in
`/workspace/runs`, log in `/workspace/runs/sweep.log`) e poi tiene vivo il pod. Se il pod
viene fermato o interrotto, al riavvio lo sweep riprende da solo. A sweep finito il pod
**resta acceso e continua a costare**: fermalo dalla console o con `runpodctl`. Senza
`SWEEP_CONFIGS` il pod parte solo con SSH, per lavorare a mano.

### Opzione B: immagine standard, senza build

Su un pod con Python 3.11+, clona il repository dentro `/workspace` e:

```bash
bash spiking-evo/runpod/setup_on_pod.sh
```

Lo script installa il pacchetto e stampa un comando `sweep` di esempio da lanciare con `nohup`.

Nota: Runpod sconsiglia di scrivere sullo stesso network volume da più pod contemporaneamente.
Per distribuire seed su più pod, dai a ogni pod il proprio volume.
