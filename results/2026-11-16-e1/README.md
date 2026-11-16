# E1 — il primo segnale: 2×2 parentela × canale, 24 seed per condizione (RunPod, 2026-11-16)

Config `e1_clones`, `e1_clones_mute`, `e1_mixed`, `e1_mixed_mute` (commit c014075): mondo di
E0 (40×40, cibo in 3 chiazze che si spostano, cibo percepito entro 2 celle, gli altri entro
6) con gruppi da 5 agenti, più lo stimolo `signal` (un'uscita "emetti", costo 0.01, sentita
dal gruppo al passo dopo entro 6 celle). Cloni: 80 genomi × 5 cloni nello stesso gruppo,
fitness = media del gruppo. Misti: 400 genomi, gruppi rimescolati a ogni episodio, fitness
individuale. "Muto" = nessuno sente il segnale. 400 agenti, 300 generazioni. Ipotesi fissate
prima del lancio in `scripts/compare_e1.py`.

Pod cpu3c 32 vCPU (EU-RO-1, 0.96 $/h), circa 1 h 57 min, costo circa 1.9 $ (stima iniziale
1.5 $: le generazioni sono state più lente del previsto; proseguito con l'ok dell'utente).
96 run su 96 completate.

| condizione | energia (ultime 20 gen.) | costo della sordità | informatività | emissioni per passo | distanza dai compagni |
|---|---|---|---|---|---|
| cloni | 404.5 | +1.38 | 0.057 | 0.120 | 15.28 |
| cloni muti | 404.9 | 0 | 0.024 | 0.152 | 15.33 |
| misti | 425.7 | +2.82 | 0.023 | 0.082 | 15.23 |
| misti muti | 424.3 | 0 | 0.029 | 0.158 | 15.23 |

Test pre-registrati (unilaterali, 20 000 permutazioni, alpha 0.05):
- **H1 (principale)** i cloni resi sordi al segnale raccolgono meno: +1.38, p = 0.080 —
  **non supportata** (tendenza)
- **H2** più energia nei cloni che nei cloni muti: −0.41, p = 0.51 — **non supportata**
- **H3** informatività più alta nei cloni che nei cloni muti: +0.033, p = 0.12 — **non
  supportata**
- **H4** informatività più alta nei cloni che nei misti: +0.034, p = 0.080 — **non
  supportata** (tendenza)
- Controllo di coerenza: costo della sordità esattamente 0 in entrambe le condizioni mute. OK.

Esplorativo (nessun test, nessuna correzione):
- **Il segnale quasi sparisce.** Emissione mediana per passo: 1.4% cloni, 3.0% cloni muti,
  1.7% misti, 2.1% misti muti; metà circa dei seed emette meno dell'1%. Il "faro" sempre
  acceso visto nel pilota dei cloni (20 generazioni) non regge: a 300 generazioni solo 2 seed
  di cloni su 24 emettono più di metà del tempo, e altrettanti (3) lo fanno nei cloni muti,
  dove il segnale non serve a nulla. I fari che restano sono deriva, non funzione.
- **Il segnale non dice "qui c'è cibo".** L'informatività supera 0.1 in 6 seed di cloni, ma
  anche in 3 seed muti e in 4 misti: emettere di più quando si mangia nasce anche senza che
  nessuno ascolti (effetto collaterale dell'attività della rete), quindi non è un indizio di
  comunicazione.
- **Chi sente lo usa un poco.** Costo della sordità positivo in 12 seed di cloni su 24 e in
  16 misti su 24 (mediana +1.27 nei misti, +0.13 nei cloni): circa 0.3% dell'energia. Il seed
  3 dei misti è un caso a parte: emette l'81% del tempo, più a digiuno che nutrito
  (informatività −0.16), e se reso sordo perde 36 di energia (circa 8%).
- **I gruppi non stanno insieme.** La distanza media dai compagni (15.2–15.3 celle in tutte
  le condizioni) è quella di punti messi a caso su una griglia 40×40 a toro (15.31). Un
  compagno è entro il raggio di ascolto (6 celle) solo circa il 7% del tempo: il segnale
  quasi sempre non arriva a nessuno.
- I misti raccolgono più energia dei cloni (+21, circa 5%): con 400 genomi la selezione ha
  più varianti su cui lavorare che con 80.

Lettura: con questo mondo **il segnale non si evolve come comunicazione**, né tra parenti né
tra estranei. Il motivo più probabile non è la parentela ma la geometria: i membri del gruppo
sono sparsi a caso e quasi mai a portata d'orecchio, e il cibo si trova bene anche da soli
(i muti raccolgono quanto i parlanti), quindi l'informazione vale poco e costa. Per E1-bis
servono condizioni in cui un'informazione serve davvero e arriva: segnale sentito in tutta
l'arena (o mondo più piccolo), cibo raro e difficile da trovare da soli, magari chiazze che
si esauriscono in fretta. Il seed 3 dei misti mostra che un uso forte del segnale è
raggiungibile.

Dati: `runs-archive/2026-11-16-e1/results.tgz` (9.1 MB, md5 37172ab8adcfcaee13f822c666291a98)
e `checkpoints.tgz` (popolazioni finali, 15.9 MB, md5 c1dbb8c51dc57c4609bbaba74b9388dc) sul
notebook, non versionati. Output completo di `compare_e1.py` in `compare_e1.txt`.
