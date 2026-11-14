# Spiking-Evo

Popolazioni di agenti guidati da reti di neuroni spiking (LIF) evolute con un algoritmo
genetico, in un mondo a griglia estendibile tramite stimoli. Stato: **milestone 1
(baseline)** completo. Specifica completa in `SPEC.md`.

## Uso

```bash
pip install -e ".[dev]"
python -m spiking_evo run spiking_evo/experiments/food_only_baseline.yaml
python -m spiking_evo run CONFIG --seed 7 --generations 100 --out runs   # override
python -m spiking_evo run CONFIG --resume                                # riprende da checkpoint.json
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
- **Simulazione batch.** Tutta la popolazione è simulata in parallelo. La propagazione degli
  spike è sparsa (gather + bincount sulle sinapsi esistenti): ~1.5 s a generazione per 200
  agenti × 1000 passi × 10 passi SNN.
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
