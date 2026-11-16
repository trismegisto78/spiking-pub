# D6 — 16 mondi con cure + 16 controllo, dono da 10 (RunPod, 2026-11-16)

Config: eco_d6_care / eco_d6_control = D5 identico (stesso mondo, stessi fondatori
`seed_populations/d5_founders.json.gz`), ma il dono vale **10** (il genitore cede 10, il figlio
riceve 8) invece di 2. Nel controllo `feed_amount: 0`. Ipotesi H1–H3 fissate prima del lancio
(`scripts/compare_d6.py`, commit 17caeee). 200 000 passi, tutti completati, nessuna estinzione.
Pod cpu3c 32 vCPU (US-MD-1, 0.96 $/h), circa 65 min, costo circa 1.05 $.

| | estinti | feed rate finale | delta feed | mortalità giovanile | dormienti alla nascita | popolazione |
|---|---|---|---|---|---|---|
| cure (dono 10) | 0/16 | 2.74 | +2.69 | 0.798 | 2.50 | 86.0 |
| controllo | 0/16 | 6.85 | +6.80 | 0.692 | 2.46 | 79.9 |

(feed rate: gesti "nutri" ogni 1000 passi-adulto, ultimi 20 000 passi)

Test pre-registrati (permutazione unilaterale):
- H1 feed_late care > control: −4.11, p = 0.921 — non supportato
- H2 delta_feed care > control: −4.11, p = 0.921 — non supportato
- H3 mortalità giovanile più bassa con cure: −0.106, p = 0.853 — non supportato

Controlli esplorativi (non pre-registrati), direzione opposta: il controllo nutre di più,
+4.11, p = 0.081; la mortalità giovanile è più alta con cure, +0.106, p = 0.146. Nessuno dei
due è significativo. Le mediane della mortalità giovanile sono quasi uguali (0.899 con cure,
0.893 controllo): la differenza tra le medie viene da 3–4 mondi di controllo con mortalità
bassa (seed 1, 13, 14, 7). Mondi con feed rate finale > 1: 5/16 con cure, 8/16 nel controllo.

Riproducibilità: il controllo D6 è la stessa config del controllo D5 con gli stessi seed e dà
gli **stessi numeri** (feed 6.846, mortalità 0.692, popolazione 79.9): la simulazione è
deterministica anche su un pod diverso.

Lettura: anche un dono 5 volte più grande non fa nascere l'accudimento. Con cure si nutre
semmai **meno** (non significativo), compatibile con un gesto che costa 10 al genitore e non
cambia la sorte del figlio: la mortalità giovanile resta intorno al 90% in quasi tutti i mondi,
con o senza dono. Il collo di bottiglia non è quanta energia passa, ma il fatto che i piccoli
muoiono comunque (pericoli e fame, vedi `scripts/diagnose_juveniles.py`).

Dati: `runs-archive/2026-11-16-eco-d6/results.tgz` sul notebook (non versionato, 783 KB,
md5 11fa69fb716b53c80a037ec558bd5982). Output completo di `compare_d6.py` in `compare_d6.txt`.
