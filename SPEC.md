# Spiking-Evo: ambiente evolutivo per reti di neuroni spiking

## Obiettivo

Simulare popolazioni di agenti, ciascuno guidato da una rete di neuroni spiking (SNN) il cui genoma evolve con un algoritmo genetico, in un ambiente che può essere arricchito nel tempo con nuovi stimoli, risorse e pericoli senza riscrivere il core.

Il progetto deve essere pensato per esperimenti di ricerca: ogni run riproducibile, ogni parametro configurabile, ogni risultato misurabile.

## Principi di design

1. **Il core non conosce gli stimoli.** Il motore di simulazione non deve contenere logica specifica di cibo, pericolo o comunicazione. Tutto ciò che influenza il mondo o i sensori passa da un'interfaccia `Stimulus`.
2. **Riproducibilità prima di tutto.** Stesso seed + stessa config = stessa identica traiettoria. Il seed globale deriva sotto-seed per ogni componente stocastico.
3. **Configurazione dichiarativa.** Ogni esperimento è un file YAML/TOML. Nessun parametro magico nel codice.
4. **Osservabilità.** Ogni generazione produce metriche su disco (CSV o Parquet) e opzionalmente snapshot di genomi migliori.
5. **Semplice prima, complesso dopo.** Il primo milestone deve girare su un solo tipo di cibo in una griglia piccola.

## Architettura

```
spiking_evo/
├── config/            # schema e loader delle configurazioni (dataclass o pydantic)
├── core/
│   ├── world.py       # griglia, agenti, step di simulazione, ordine di aggiornamento
│   ├── agent.py       # stato fisico di un agente (posizione, energia, orientamento)
│   └── clock.py       # gestione dei passi temporali e del dt della SNN
├── snn/
│   ├── neuron.py      # leaky integrate-and-fire (LIF), parametri per neurone
│   ├── network.py     # rete: neuroni, sinapsi, ritardi, propagazione spike
│   └── encoders.py    # sensori -> treni di spike (rate coding e/o temporal coding)
├── stimuli/
│   ├── base.py        # interfaccia Stimulus (vedi sotto)
│   ├── food.py        # cibo con rigenerazione
│   └── registry.py    # registrazione e caricamento degli stimoli dalla config
├── genetics/
│   ├── genome.py      # rappresentazione genetica della rete
│   ├── operators.py   # mutazione, crossover, selezione
│   ├── speciation.py  # (milestone 3) raggruppamento per similarità genetica
│   └── population.py  # ciclo generazionale
├── fitness/
│   └── evaluator.py   # calcolo fitness da energia, attività, comportamento
├── metrics/
│   └── logger.py      # scrittura metriche per generazione e snapshot
├── experiments/       # file di config dei singoli esperimenti
└── tests/
```

## Interfaccia per gli stimoli (punto di estensione)

Ogni stimolo implementa la stessa interfaccia, così nuovi elementi si aggiungono senza toccare il core:

```python
class Stimulus(Protocol):
    name: str

    def reset(self, world: World, rng: np.random.Generator) -> None:
        """Inizializza lo stimolo all'inizio di un episodio."""

    def step(self, world: World, t: int) -> None:
        """Aggiorna lo stato dello stimolo (rigenerazione, movimento, ecc.)."""

    def sense(self, agent: Agent, world: World) -> np.ndarray:
        """Restituisce un vettore di intensità percepite dall'agente."""

    def interact(self, agent: Agent, world: World) -> float:
        """Effetto sull'agente (es. energia guadagnata o persa). Ritorna il delta energia."""
```

Ogni stimolo dichiara nella config quali canali sensoriali occupa, così la dimensione dell'input della rete si calcola automaticamente.

Stimoli previsti in ordine di sviluppo:

- **Milestone 1:** `FoodStimulus` con cibo fisso o a rigenerazione lenta.
- **Milestone 2:** `HazardStimulus` (pericolo che sottrae energia), `FoodPatchStimulus` (cibo raggruppato in zone).
- **Milestone 3:** `SignalChannel` (canale in cui gli agenti emettono valori visibili ai vicini, base per la comunicazione).
- **Futuro:** cambiamenti ambientali nel tempo (stagioni, cibo che si sposta), agenti concorrenti.

## Modello neurale

- **Neurone:** LIF con parametri `tau_m`, `v_thresh`, `v_reset`, `refractory`. Parametri per neurone, non globali, così l'evoluzione può modificarli.
- **Sinapsi:** peso `w` (può essere negativo per inibizione), ritardo intero in passi di simulazione.
- **Input:** i sensori producono correnti o treni di spike tramite encoder. Il tipo di encoder è configurabile.
- **Output:** attività di un sottoinsieme di neuroni dedicati a motori (es. due neuroni per l'avanti-indietro, due per la rotazione), decodificata come rate sulla finestra temporale.
- **Dimensioni:** la rete ha un numero variabile di neuroni nascosti, deciso dal genoma, con un limite massimo configurabile.

## Genoma

Il genoma descrive la rete in forma diretta o indiretta, selezionabile dalla config:

- **Diretta:** lista di neuroni (parametri) e lista di sinapsi (sorgente, destinazione, peso, ritardo).
- **Indiretta (milestone 3):** regole di crescita che generano la rete a partire da un seme, per favorire moduli riusabili.

Vincoli:

- Il genoma deve essere serializzabile (JSON o pickle versionato) per poter riprendere un esperimento.
- Gli operatori genetici non devono mai produrre reti invalide: i pesi vengono clippati, le sinapsi verso neuroni inesistenti scartate.

## Fitness

La fitness è una combinazione configurabile di termini:

- energia accumulata durante la vita dell'agente;
- **penalità** per firing rate medio sopra una soglia (evita reti saturate);
- **penalità** per silenzio totale (evita reti morte);
- bonus opzionale per esplorazione (celle visitate distinte) per contrastare la convergenza prematura.

Ogni termine ha un peso nella config e può essere disattivato. Ogni termine viene loggato separatamente, così si vede quale domina.

## Algoritmo genetico

- Popolazione di dimensione configurabile (default 200).
- Selezione per torneo con dimensione configurabile.
- Crossover opzionale, mutazione sempre attiva su pesi, soglie, presenza di sinapsi e numero di neuroni.
- Elitismo: i migliori N individui passano invariati.
- Milestone 3: speciazione con distanza genetica, per mantenere diversità.
- Il seed di ogni generazione è derivato dal seed globale, così ogni generazione è riproducibile in isolamento.

## Metriche e output

Per ogni generazione, scritte in `runs/<nome_esperimento>/<seed>/`:

- `generations.csv`: fitness media, massima, minima, varianza, numero di specie (se attive), firing rate medio, energia media.
- `terms.csv`: contributo di ogni termine di fitness.
- `snapshots/gen_XXXX.json`: genoma del migliore, ogni N generazioni.
- `config.yaml`: copia della config usata, più il commit hash del repo.

## Configurazione di esempio

```yaml
experiment: food_only_baseline
seed: 42
world:
  width: 40
  height: 40
  steps_per_episode: 1000
stimuli:
  - type: food
    count: 60
    regrow_rate: 0.01
network:
  encoder: rate
  hidden_max: 30
  dt_ms: 1.0
  neuron: { tau_m: 20.0, v_thresh: 1.0, v_reset: 0.0, refractory_ms: 2.0 }
genetics:
  population: 200
  generations: 300
  tournament_size: 3
  elitism: 5
  mutation: { weight: 0.1, threshold: 0.05, add_synapse: 0.02, remove_synapse: 0.01 }
fitness:
  energy: 1.0
  firing_penalty: { target_rate_hz: 20, weight: 0.5 }
  silence_penalty: { weight: 0.5 }
  exploration: { weight: 0.0 }
```

## Milestone

1. **Baseline:** griglia 40x40, solo cibo fisso, rete diretta, GA senza speciazione. Deve girare e produrre `generations.csv`. Criterio di successo: la fitness media cresce in modo misurabile in 100 generazioni.
2. **Ambiente ricco:** aggiungere `HazardStimulus` e `FoodPatchStimulus`. Criterio: gli agenti distinguono zone di cibo e pericolo, misurato con un test comportamentale.
3. **Diversità e comunicazione:** speciazione, genoma indiretto, `SignalChannel`.
4. **Analisi:** notebook o script che legge `runs/` e produce grafici di convergenza, mappe di attività e confronto tra esperimenti.

## Requisiti tecnici

- Python 3.11+
- NumPy per la simulazione; nessuna dipendenza GPU nel baseline.
- Brian2 o Norse opzionali, come backend alternativi per la SNN. Il backend nativo NumPy è quello di riferimento.
- `pytest` per i test, `ruff` per il linting.
- Ogni modulo in `core/`, `snn/` e `genetics/` deve avere test unitari. Il test minimo: un neurone LIF con input costante spara alla frequenza attesa; una rete con una sola sinapsi propaga lo spike con il ritardo corretto; due run con stesso seed producono lo stesso genoma finale.

## Istruzioni per chi implementa

- Implementa il milestone 1 per intero prima di toccare il milestone 2.
- Non introdurre dipendenze nuove senza motivarlo nel messaggio di commit.
- Ogni nuovo stimolo va aggiunto solo creando un file in `stimuli/` e registrandolo nel registry: nessuna modifica a `core/` per un nuovo stimolo. Se serve modificare il core, fermarsi e documentare perché.
- Scrivere prima il test, poi il codice, quando il comportamento è deducibile dalla specifica.
- Mantenere le metriche attive per default: un esperimento senza log non conta come esperimento.
