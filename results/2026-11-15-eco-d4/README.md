# D4 "naturale" — 32 mondi con cure + 32 controllo (RunPod, 2026-11-15)

Config: eco_d4_care / eco_d4_control (cibo 40, 8 zone fertili, capacità 300, 200 000 passi).
Fermato a mano dopo ~2 h 40 (costo ~2.7 $): 11 mondi di controllo erano tra 158 000 e
194 000 passi; analizzati sugli ultimi 20 000 passi raggiunti (`compare_d4.py runs 150000`).
Le estinzioni sono riconoscibili: finiscono tutte entro 68 000 passi.

| | estinti | feed rate finale (per 1000 passi-adulto) | delta feed | mortalità giovanile | neuroni dormienti alla nascita |
|---|---|---|---|---|---|
| cure | 8/32 | 13.0 | -34.5 | 0.026 | 3.30 |
| controllo | 4/32 | 109.0 | -27.4 | 0.029 | 3.51 |

Test pre-registrati (permutazione, unilaterali):
- H1 nutrire è più frequente con cure: **smentito**, diff -95.9, p = 1.000 (è vero il contrario).
- H2 nutrire cresce di più con cure: non supportato, p = 0.584.
- H3 meno piccoli morti con cure: non supportato, p = 0.339.

Lettura: dove nutrire costa energia al donatore, la selezione lo *elimina* (nel 20/24 dei mondi
sopravvissuti scende quasi a zero); dove non costa nulla (controllo) il gesto deriva liberamente.
Il dono non porta un vantaggio genetico perché con kin_only false si nutrono piccoli qualsiasi e
la popolazione è sempre al tetto (300): un piccolo salvato non aumenta i discendenti propri.
