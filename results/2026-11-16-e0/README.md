# E0 — vedere gli altri: 24 seed sociali + 24 controllo (RunPod, 2026-11-16)

Primo passo verso la comunicazione. Config `e0_social` / `e0_control`: gruppi da 10 agenti
per mondo (arene a gruppi, rimescolate a ogni episodio), griglia 40×40, 60 cibi in 3 chiazze
(raggio 1.5) che si spostano ogni 250 passi, cibo percepito solo entro 2 celle; nella
condizione sociale gli altri membri del gruppo si percepiscono entro 6 celle (stimolo
`others`), nel controllo gli stessi ingressi restano sempre a 0. Fitness individuale
(energia raccolta). 200 generazioni, popolazione 200. Ipotesi fissate prima del lancio in
`scripts/compare_e0.py` (commit 38fc01b).

Pod cpu3c 32 vCPU (EU-RO-1, 0.96 $/h), circa 23 min, costo circa 0.37 $. 48 run su 48
completate.

| | energia (ultime 20 gen.) | costo della sordità agli altri |
|---|---|---|
| sociale | 301.3 | +2.36 |
| controllo | 294.1 | 0.00 (atteso: controllo di coerenza) |

Test pre-registrati (permutazione unilaterale, alpha 0.05):
- **H1** più energia con il senso sociale: +7.1 (+2.4%), p = 0.218 — **non supportata**
- **H2** la popolazione sociale finale raccoglie meno se resa sorda agli altri: +2.36,
  p = 0.010 — **supportata**

Esplorativo (non pre-registrato, nessuna correzione per confronti multipli):

| generazione | 0 | 9 | 24 | 49 | 99 | 149 | 199 |
|---|---|---|---|---|---|---|---|
| energia, sociale | 18 | 183 | 235 | 266 | 285 | 294 | 303 |
| energia, controllo | 14 | 172 | 217 | 250 | 278 | 288 | 296 |
| vivi a fine episodio, sociale | 0.52 | 0.97 | 0.98 | 0.98 | 0.97 | 0.98 | 0.98 |
| vivi a fine episodio, controllo | 0.73 | 0.93 | 0.94 | 0.94 | 0.93 | 0.95 | 0.95 |

- Nelle prime 50 generazioni la condizione sociale raccoglie di più (+15, p = 0.03) e
  sopravvive di più (+3.5 punti, p < 0.001); a fine run il vantaggio di energia si riduce
  (+7, p = 0.21) ma la sopravvivenza resta più alta (+1.9 punti, p < 0.001).
- Il costo della sordità varia molto tra seed: da −6.6 a +13.6; 17 seed su 24 positivi.
  Il seed 8 dipende di più dagli altri (+13.6, circa 6% della sua energia).

Lettura: gli agenti **imparano a usare la presenza degli altri** come indizio (H2), e questo
accelera l'evoluzione e protegge dalla morte, ma alla lunga il controllo recupera quasi tutto
il vantaggio di energia con la sola ricerca individuale. L'informazione sociale qui è un
*indizio involontario* (dove sono gli altri), che porta con sé anche competizione: seguire
qualcuno significa arrivare dove sta già mangiando. È il punto di partenza naturale per E1:
un segnale che l'individuo *sceglie* di emettere.

Dati: `runs-archive/2026-11-16-e0/results.tgz` (2.1 MB, md5 dafdf1f9d4d981e031961b327c1c85aa)
e `checkpoints.tgz` (popolazioni finali, 4.0 MB, md5 fc95f0f324108e329ce596ee16623485) sul
notebook, non versionati. Output completo di `compare_e0.py` in `compare_e0.txt`.
