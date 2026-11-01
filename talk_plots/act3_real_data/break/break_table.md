# Predicted crack onset vs the real first crack

Real first crack (test): first load drop at the peak, 4,755 N at 2.51 mm crosshead (digitized curve; the machine report says 4.80 kN at 2.56 mm). Where: the lower inner bore (face f5, near (-14.56, -13.22, -4.31) mm), observed by the team on the broken part (no photo in the repo).
Criteria: PEEQ >= 0.043 (dogbone plastic strain at break), s1 >= 50.48 MPa (dogbone UTS: 'predicted crack'), s1 >= 85% of it = 42.9 MPa ('potential crack': a margin for print defects and layer lines; the threshold was chosen after seeing this test). Raw: the single highest node; robust: reached by >= 3 nodes within 3.0 mm. Travel = each case's own crosshead mapping.

| case | PEEQ, raw | PEEQ, robust | s1 >= 100 %, raw | s1 >= 100 %, robust | potential: s1 >= 85 %, raw | site matches the real crack (observed: lower inner bore)? |
|---|---|---|---|---|---|---|
| before: what we started with (2,937 nodes, 33 % off) | 0.98 mm, 2,535 N | 1.05 mm, 2,633 N | 1.76 mm, 3,162 N | 1.86 mm, 3,193 N | 1.50 mm, 3,045 N | 85 %: no (prong inner side); 100 %: no (prong inner side) |
| stage2: learned test conditions, default mesh (2,937 nodes, 7.8 % off) | 1.57 mm, 3,689 N | 1.81 mm, 4,145 N | no break by 2.79 mm (42 MPa at the real crack) | no break by 2.79 mm (31.3 MPa at the real crack) | 2.56 mm, 5,226 N | 85 %: no (outer side); 100 %: no (outer side, most stressed) |
| stage3: learned conditions, pre-trained mesh (6,430 nodes, 4.6 % off) | 1.22 mm, 2,837 N | 1.37 mm, 3,181 N | 2.58 mm, 5,045 N | no break by 2.79 mm (41.3 MPa at the real crack) | 2.18 mm, 4,570 N | 85 %: no (nut-pocket corner); 100 %: no (nut-pocket corner) |
| after: trained on the real test: most accurate mesh (8,753 nodes, 2.0 % off) | 1.16 mm, 2,659 N | 1.31 mm, 2,969 N | no break by 2.78 mm (43 MPa at the real crack) | no break by 2.78 mm (40.8 MPa at the real crack) | 2.51 mm, 4,814 N | 85 %: yes (lower inner bore); 100 %: yes (lower inner bore, most stressed) |

Sites (first node over the threshold, else the most stressed at the end; undeformed xyz mm, faces within 2 mm):

- before, PEEQ, raw: site [20.35, 25.74, 6.62] mm, f41 / f42 (nut pocket) (nut pocket); value at the real crack travel 0.249
- before, PEEQ, robust: site [22.03, 24.31, 5.79] mm, f42 (nut pocket) (nut pocket); value at the real crack travel 0.215
- before, s1 >= 100 %, raw: site [12.93, 26.27, 8.1] mm, f13 / f16 (flat side, prong inner side) (prong inner side); value at the real crack travel 68.1
- before, s1 >= 100 %, robust: site [-12.93, 26.01, 5.73] mm, f14 (prong inner side) (prong inner side); value at the real crack travel 66.9
- before, potential: s1 >= 85 %, raw: site [12.93, 26.27, 8.1] mm, f13 / f16 (flat side, prong inner side) (prong inner side); value at the real crack travel 68.1
- before, potential: s1 >= 85 %, robust: site [-12.93, 26.01, 5.73] mm, f14 (prong inner side) (prong inner side); value at the real crack travel 66.9
- stage2, PEEQ, raw: site [20.35, 25.74, 6.62] mm, f41 / f42 (nut pocket) (nut pocket); value at the real crack travel 0.134
- stage2, PEEQ, robust: site [22.03, 24.31, 5.79] mm, f42 (nut pocket) (nut pocket); value at the real crack travel 0.097
- stage2, s1 >= 100 %, raw: max at the end [-27.61, 2.98, -8.1] mm, f6 / f12 (flat side, outer side) (outer side); value at the real crack travel 42; most stressed at the real crack [-27.61, 2.98, -8.1] mm, f6 / f12 (flat side, outer side)
- stage2, s1 >= 100 %, robust: max at the end [-15.96, 22.37, 0.0] mm, f36 / f55 (prong hole) (prong hole); value at the real crack travel 31.3; most stressed at the real crack [-15.96, 22.37, 0.0] mm, f36 / f55 (prong hole)
- stage2, potential: s1 >= 85 %, raw: site [-27.61, 2.98, -8.1] mm, f6 / f12 (flat side, outer side) (outer side); value at the real crack travel 42
- stage2, potential: s1 >= 85 %, robust: max at the end [-15.96, 22.37, 0.0] mm, f36 / f55 (prong hole) (prong hole); value at the real crack travel 31.3; most stressed at the real crack [-15.96, 22.37, 0.0] mm, f36 / f55 (prong hole)
- stage3, PEEQ, raw: site [22.59, 25.74, -6.62] mm, f8 / f12 / f38 / f39 (flat side, nut pocket, prong outer side) (nut-pocket corner); value at the real crack travel 0.226
- stage3, PEEQ, robust: site [22.59, 25.74, -6.62] mm, f8 / f12 / f38 / f39 (flat side, nut pocket, prong outer side) (nut-pocket corner); value at the real crack travel 0.18
- stage3, s1 >= 100 %, raw: site [23.72, 25.83, 8.1] mm, f8 / f13 / f41 / f42 (flat side, nut pocket, prong outer side) (nut-pocket corner); value at the real crack travel 49.4
- stage3, s1 >= 100 %, robust: max at the end [14.12, -13.69, -2.24] mm, f5 (inner bore) (lower inner bore); value at the real crack travel 41.3; most stressed at the real crack [14.12, -13.69, -2.24] mm, f5 (inner bore)
- stage3, potential: s1 >= 85 %, raw: site [23.72, 25.83, 8.1] mm, f8 / f13 / f41 / f42 (flat side, nut pocket, prong outer side) (nut-pocket corner); value at the real crack travel 49.4
- stage3, potential: s1 >= 85 %, robust: site [14.12, -13.69, -2.24] mm, f5 (inner bore) (lower inner bore); value at the real crack travel 41.3
- after, PEEQ, raw: site [21.19, 25.74, -6.62] mm, f12 / f38 / f39 (flat side, nut pocket) (nut pocket); value at the real crack travel 0.292
- after, PEEQ, robust: site [21.19, 25.74, 6.62] mm, f13 / f41 / f42 (flat side, nut pocket) (nut pocket); value at the real crack travel 0.2
- after, s1 >= 100 %, raw: max at the end [-14.56, -13.22, -4.31] mm, f5 (inner bore) (lower inner bore); value at the real crack travel 43; most stressed at the real crack [-14.56, -13.22, -4.31] mm, f5 (inner bore)
- after, s1 >= 100 %, robust: max at the end [-14.75, -13.01, -2.37] mm, f5 (inner bore) (lower inner bore); value at the real crack travel 40.8; most stressed at the real crack [-14.75, -13.01, -2.37] mm, f5 (inner bore)
- after, potential: s1 >= 85 %, raw: site [-14.56, -13.22, -4.31] mm, f5 (inner bore) (lower inner bore); value at the real crack travel 43
- after, potential: s1 >= 85 %, robust: site [-14.75, -13.01, -2.37] mm, f5 (inner bore) (lower inner bore); value at the real crack travel 40.8

Potential crack (s1 >= 85 %), robust: before 1.63 mm, 3,108 N; stage2 no potential crack by 2.79 mm (31.3 MPa at the real crack); stage3 2.62 mm, 5,085 N; after 2.65 mm, 4,950 N


Mesh sensitivity (learned conditions, same ramp; onset travel by each criterion; value at the real crack travel in brackets):

| mesh | nodes | PEEQ raw | PEEQ robust | s1 raw (MPa) | s1 robust (MPa) | s1 >= 85 %, raw (site) |
|---|---|---|---|---|---|---|
| automesher (gmsh defaults) | 2,937 | 1.57 mm [0.134] | 1.81 mm [0.097] | none [42] | none [31.3] | 2.56 mm (outer side) |
| pre-trained swarm mesh | 6,430 | 1.22 mm [0.226] | 1.37 mm [0.18] | 2.58 mm [49.4] | none [41.3] | 2.18 mm (nut-pocket corner) |
| uniform 4 mm | 9,506 | 1.11 mm [0.306] | 1.48 mm [0.165] | none [42.4] | none [40.9] | 2.54 mm (lower inner bore) |
| uniform 2.5 mm | 24,194 | 1.24 mm [0.226] | 1.39 mm [0.18] | none [41.9] | none [40.5] | 2.58 mm (lower inner bore) |
| uniform 2 mm | 41,186 | 1.11 mm [0.271] | 1.25 mm [0.241] | 2.71 mm [45.8] | none [40.2] | 2.38 mm (nut-pocket corner) |
