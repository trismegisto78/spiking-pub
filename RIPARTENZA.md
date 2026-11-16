# Ripartenza: riepilogo per una nuova sessione (16/11/2026)

Documento breve per riprendere il lavoro da zero. I dettagli completi sono in
`STATO_LAVORI.md` (tutti i risultati, la mappa del codice, le decisioni prese) e in `results/`.

## 1. Il progetto in due righe

**Spiking-Evo**: creature con cervello a neuroni *spiking* (LIF) vivono su una griglia ed
evolvono con un algoritmo genetico. Domanda di fondo: quali capacità (evitare pericoli,
coltivare, crescere il cervello, accudire i piccoli) **emergono da sole**, perché servono,
senza essere imposte. Filosofia dell'utente: *non forzare la biologia*. Se una linea si
estingue è un risultato; il mondo deve rendere *vantaggioso* un comportamento, non
obbligarlo.

## 2. Dove sta il codice

| cosa | dove |
|---|---|
| repository principale (privato) | `trismegisto78/spiking-evo`, ramo di lavoro **`claude/plasticity`** (contiene tutto) |
| copia pubblica per i pod RunPod | `trismegisto78/spiking-pub`, ramo `main` (aggiornare con `git push https://github.com/trismegisto78/spiking-pub HEAD:main`) |
| Python | 3.11 o superiore; `pip install -e ".[dev,analysis]"`; `python -m pytest` (150 test, circa 10 s) |

Comandi principali (`python -m spiking_evo ...`): `run` (evoluzione a generazioni), `sweep`
(più seed), `ecosystem` (mondo continuo con nascite e morti), `analyze`, `behaviour`
(test con un senso accecato), `cpus`. Variabile `SPIKING_EVO_OUT` = cartella dei risultati.

Pezzi chiave:
- `spiking_evo/stimuli/`: plugin del mondo: `food`, `hazard` (danni a livelli), `terrain`
  (azione "agisci": semina su suolo fertile nascosto oppure avvelena; varietà di raccolto;
  `fallow_time` = riposo del suolo dopo un raccolto), `season` (stagioni con costo di vivere),
  `body`, `kin` (sensi figlio/genitore, azione "nutri", `kin_only`).
- `spiking_evo/snn/`: rete, plasticità (STDP con ricompensa), sviluppo (neuroni dormienti che si
  svegliano investendo energia).
- `spiking_evo/ecosystem/sim.py`: ecosistema (geni di storia di vita: soglia di riproduzione,
  dote ai figli, numero di figli, età adulta).
- `scripts/compare_*.py`: analisi con ipotesi **fissate prima** di lanciare (test di
  permutazione). `scripts/diagnose_juveniles.py`: bilancio energetico per vita di piccoli e
  adulti.
- `runpod/eco_job.sh`, `runpod/sweep_job.sh`: job per i pod (variabili CONFIGS, SEEDS, COMPARE).

## 3. Cosa abbiamo scoperto finora

| tappa | esito |
|---|---|
| baseline solo cibo (8 seed, notebook) | strategia evoluta = **veicolo di Braitenberg** (gira verso il lato col cibo); plateau ~550 dalla gen 200 |
| Milestone 1–2 (cibo, pericoli, terreno, raccolti, stagioni) | le creature imparano a evitare i pericoli; l'agricoltura emerge solo se il terreno è abbastanza facile |
| A. continuità tra ambienti | **aiuta molto**: chi arriva con una storia evolutiva fa ~6 volte meglio di chi parte da zero |
| B. plasticità (STDP con ricompensa) | **non aiuta**; l'evoluzione abbassa la velocità di apprendimento |
| C. cervello che nasce piccolo e cresce | con 8 seed sembrava aiutare; **con 24 seed la differenza sparisce** (lezione: servono almeno 24 seed) |
| D1–D3 ecosistema e accudimento | l'accudimento non emerge |
| D4 "naturale" (32+32 mondi) | nutrire **viene eliminato** (costa, si nutre chiunque, popolazione sempre al tetto) |
| D5 (16+16): dono solo ai figli, suolo che riposa, cibo 500, fondatori evoluti nel mondo D5 | mondo sano (popolazione 35–117, nessuna estinzione), ma **nessuna differenza** tra cure e controllo (p ≈ 0.47–0.62) |
| D6 (16+16): come D5 ma dono da 10 | ancora **nessuna differenza** (H1–H3 p = 0.92/0.92/0.85); con cure si nutre semmai meno; mortalità giovanile circa 0.9 ovunque |

Diagnosi dopo D5 (`scripts/diagnose_juveniles.py`): i piccoli muoiono di **pericoli** (−16…−22
energia, quanto tutta la loro dote) e di **fame**; l'aiuto dei genitori vale +0.06…+0.13. Il
dono (2 energia, 1.6 ricevuti) è troppo piccolo per cambiare il destino di un figlio. Con un dono
da 10 (test 8, 20 000 passi) il mondo regge, ma i fondatori nutrono ancora di rado: solo una
simulazione lunga può dire se la selezione lo fa crescere.

## 4. Prossimo passo

D6 fatto (vedi sopra e `results/2026-11-16-eco-d6/`): il dono più grande non basta. Da
decidere con l'utente. Il dato più forte è che i piccoli muoiono quasi tutti (circa 90%)
qualunque cosa facciano i genitori, quindi ha senso capire prima **perché** (pericoli? fame?
cervello troppo piccolo?) con test brevi, e solo dopo cambiare il mondo.

## 5. Richieste aperte dell'utente

- **Far girare sul suo notebook** (NOTEBOOK-ALE: Windows 11, Python 3.12, **16 core logici**;
  i 150 test passano lì, verificato il 16/11/2026 con il commit "Baseline with 8 seeds…").
  Tempi **misurati** il 16/11: un mondo da solo circa 8 s ogni 1000 passi, ma con 16 mondi in
  parallelo circa 75 s ogni 1000 passi (8 core fisici): 8+8 da 200 000 passi in circa 3.5 h.
  Per esperimenti lunghi usare RunPod; sul notebook solo test brevi. Comando:
  `python -m spiking_evo ecosystem spiking_evo/experiments/eco_d6_care.yaml --seed N` (un
  processo per mondo), oppure uno script che li lanci in parallelo come `runpod/eco_job.sh`.
  Dalle sessioni cloud le sue cartelle (`spike`, …) non risultavano utilizzabili: o si aggiorna
  l'app Claude sul notebook, o si apre la sessione direttamente lì (consigliato per GUI e run
  locali).
- **GUI** per guardare l'evoluzione: comando `python -m spiking_evo watch` che apre nel browser
  (senza dipendenze extra: `http.server` + canvas JS)
  1. il mondo dal vivo: griglia, adulti e piccoli con colori diversi, linee genitore-figlio,
     cibo, pericoli, campi, veleno, lampi quando qualcuno nutre, stagioni, pausa/velocità,
     clic su una creatura per vederne lo stato e i geni;
  2. un cruscotto che legge gli `ecosystem.csv` dei mondi in corso: popolazione, gesti
     "nutri" con cure e senza, mortalità dei piccoli, cervello alla nascita, geni medi.

## 6. Come lavorare con l'utente

- Rispondere **in italiano**, con parole semplici e onestà (anche i risultati negativi).
- **Test piccoli, da 5–10 minuti, uno alla volta**: l'utente non vuole aspettare mezz'ora. Per
  calcoli lunghi usare RunPod, **dire il costo prima** di creare un pod e **spegnerlo** a fine
  lavoro. Pod tipico: CPU `cpu3c` 32 vCPU, 0.96 $/h, US-MD-1, immagine `python:3.12`; i
  risultati si leggono con `exec-pod`.
- Il token GitHub **non va mai** chiesto né scritto in chat.
- Ipotesi e test **fissati prima** di lanciare; contare le estinzioni come esito.
- Committare e pushare su `claude/plasticity` (e su spiking-pub quando serve ai pod).

## 7. Messaggio pronto per la nuova sessione

> Leggi `RIPARTENZA.md` e poi `STATO_LAVORI.md` del repo `trismegisto78/spiking-evo`, ramo
> `claude/plasticity`. Rispondimi in italiano. Vorrei [costruire la GUI `watch` / lanciare D6
> sul mio notebook / lanciare D6 su RunPod]. Lavora a test piccoli da 5–10 minuti e dimmi i
> costi prima di usare RunPod.
