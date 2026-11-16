# D5 — 16 mondi con cure + 16 controllo (RunPod, 2026-11-16)

Config: eco_d5_care / eco_d5_control = D4 + dono solo ai propri figli (kin_only) + riposo del
terreno 2000 passi + cibo 500 + fondatori evoluti 300 generazioni nel mondo D5
(`seed_populations/d5_founders.json.gz`). Capacità 300, mai raggiunta: la popolazione si
regola da sola (35–117). 200 000 passi, tutti completati, nessuna estinzione.
Durata ~70 min, costo ~1.1 $.

| | estinti | feed rate finale | delta feed | mortalità giovanile | dormienti alla nascita | popolazione |
|---|---|---|---|---|---|---|
| cure | 0/16 | 7.19 | +7.12 | 0.725 | 2.18 | 79.3 |
| controllo | 0/16 | 6.85 | +6.80 | 0.692 | 2.46 | 79.9 |

(feed rate: gesti "nutri" ogni 1000 passi-adulto, ultimi 20 000 passi)

Test pre-registrati (permutazione unilaterale):
- H1 feed_late care > control: +0.34, p = 0.465 — non supportato
- H2 delta_feed care > control: +0.33, p = 0.471 — non supportato
- H3 mortalità giovanile più bassa con cure: -0.033, p = 0.619 — non supportato

Lettura: nessuna differenza. In entrambe le condizioni il gesto "nutri" cresce da ~0 a 10–40
in circa metà dei mondi: deriva (o effetto collaterale), non selezione, visto che nel
controllo non trasferisce nulla. Rispetto a D4 non viene più eliminato (il dono ai soli figli
non è più uno spreco), ma nemmeno premiato. Probabile motivo: il dono è energeticamente
irrilevante (2 energia, 1.6 ricevuti, rari) mentre la mortalità giovanile è altissima
(70–90% in molti mondi): il piccolo muore per motivi che qualche boccone non cambia.

Dati per mondo: vedi output completo sotto.
```
care:    1:22.0  2:18.6  3:25.0  4:0.24  5:9.2  6:0.03  7:0.05  8:0.10  9:0.02  10:0.02  11:0.03  12:0.04  13:39.3  14:0.00  15:0.51  16:0.00  (feed_late)
control: 1:0.01  2:8.7   3:19.5  4:0.30  5:10.6 6:20.5  7:0.01  8:0.63  9:0.02  10:11.7  11:10.5  12:26.0  13:0.18  14:0.02  15:1.03  16:0.00  (feed_late)
```
