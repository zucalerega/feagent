# code_aster vendored tests

Generated from `manifest.json` (upstream commit 7cb48262af). 253 SSLV/SDLV variants parsed; 178 have independent references; the 80 below are vendored in `astest/`. 'code_aster vs ref' is the median |VALE_CALC - VALE_REFE| / |VALE_REFE| over the test's independent references: how close code_aster itself gets on the test mesh.

| Test | Title (from the command file) | Model | Operators | Independent refs (located) | code_aster vs ref, median |
|---|---|---|---|---|---|
| sdlv100a | POUTRE ELANCEE DE SECTION RECTANGULAIRE VARIABLE | 3D | CALC_MODES | 10 (0) | 0.18 % |
| sdlv100b | POUTRE ELANCEE DE SECTION RECTANGULAIRE VARIABLE | 3D | CALC_MODES | 10 (0) | 0.15 % |
| sdlv111a | RESEAU PERIODIQUE DE POUTRES BAIGNE PAR UN FLUIDE | 3D_FAISCEAU | CALC_MODES | 16 (0) | 0.3 % |
| sdlv111b | RESEAU PERIODIQUE DE POUTRES BAIGNE PAR UN FLUIDE | 3D_FAISCEAU | CALC_MODES | 7 (0) | 0 % |
| sdlv126h | BUT DE LA FAMILLE DE TEST | 3D | DYNA_VIBRA | 1 (0) | - |
| sdlv126i | BUT DE LA FAMILLE DE TEST | 3D | DYNA_VIBRA | 1 (0) | - |
| sdlv126j | BUT DE LA FAMILLE DE TEST | 3D | DYNA_VIBRA | 1 (0) | - |
| sdlv127a | Fichier de commande onde plane inclinee (sur un par.ede de 200 par 400 | 3D, 3D_ABSO | DYNA_VIBRA | 12 (0) | 0.51 % |
| sdlv127c | Fichier de commande onde plane inclinee (sur un par.ede de 200 par 400 | 3D, 3D_ABSO, DIS_T | DYNA_VIBRA | 12 (0) | 0.53 % |
| sdlv127e | Fichier de commande onde plane inclinee (sur un par.ede de 200 par 400 | 3D, 3D_ABSO | DYNA_VIBRA | 12 (0) | 0.5 % |
| sdlv127f | Fichier de commande onde plane inclinee (sur un par.ede de 200 par 400 | 3D, 3D_ABSO | DYNA_VIBRA | 12 (0) | 11 % |
| sdlv127m | Fichier de commande onde plane inclinee (sur un par.ede de 200 par 400 | 3D, 3D_ABSO | DYNA_VIBRA | 1 (0) | 4.8e-11 % |
| sdlv127n | Fichier de commande onde plane inclinee (sur un par.ede de 200 par 400 | 3D, 3D_ABSO | DYNA_VIBRA | 1 (0) | 1.4e-10 % |
| sdlv127o | Fichier de commande onde plane inclinee (sur un par.ede de 200 par 400 | 3D, 3D_ABSO | DYNA_VIBRA | 2 (0) | 8.5e-11 % |
| sdlv127p | Fichier de commande onde plane inclinee (sur un par.ede de 200 par 400 | 3D, 3D_ABSO | DYNA_VIBRA | 2 (0) | 1.2e-10 % |
| sdlv129a | MAILLAGE, MODELE ET MATERIAU | 3D | CALC_MODES, MECA_STATIQUE | 5 (4) | 0.0056 % |
| sdlv131a | - CALCUL DE MOYENNE DE LA DEFORMATION | 3D, DIS_T | MECA_STATIQUE | 6 (6) | - |
| sdlv131b | - CALCUL DE MOYENNE DE LA DEFORMATION | 3D, DIS_T | DYNA_VIBRA | 7 (7) | - |
| sdlv131c | - CALCUL DE MOYENNE DE LA DEFORMATION | 3D, DIS_T | DYNA_VIBRA | 6 (6) | - |
| sdlv134a | VALIDATION SOUS-STRUCTURATION CYCLIQUE | 3D | CALC_MODES | 18 (0) | - |
| sdlv301a | MODELE AVEC 160 ELEMENTS DANS LE SECTEUR DE BASE | 3D | CALC_MODES | 8 (0) | 2.3 % |
| sdlv301b | MODELE AVEC 160 ELEMENTS DANS LE SECTEUR DE BASE | 3D | CALC_MODES | 8 (0) | 2.3 % |
| sdlv302c | AMORTISSEMENT VISQUEUX PROPORTIONNEL | 3D | CALC_MODES | 4 (4) | 0 % |
| sdlv401a | ELEMENTS  : HEXA8 | 3D | CALC_MODES | 10 (0) | 0.00017 % |
| sdlv401b | ELEMENTS  : HEXA20 | 3D | CALC_MODES | 10 (0) | 0.025 % |
| sslv01a | unites: MPa, mm | 3D, 3D_INTERFACE_S | STAT_NON_LINE | 3 (3) | - |
| sslv04a | CYLINDRE CREUX EN CONTRAINTES PLANES | 3D | MECA_STATIQUE, STAT_NON_LINE | 123 (123) | 0.42 % |
| sslv04b | CYLINDRE CREUX EN CONTRAINTES PLANES | 3D | MECA_STATIQUE | 77 (0) | 0.28 % |
| sslv04c | CYLINDRE CREUX EN CONTRAINTES PLANES | 3D | MECA_STATIQUE | 73 (73) | 2.1 % |
| sslv04d | CYLINDRE CREUX EN CONTRAINTES PLANES | 3D | MECA_STATIQUE | 66 (0) | 0.33 % |
| sslv04l | CYLINDRE CREUX EN CONTRAINTES PLANES | 3D | MECA_STATIQUE | 40 (40) | 4.8 % |
| sslv04m | CYLINDRE CREUX EN CONTRAINTES PLANES | 3D | MECA_STATIQUE | 40 (40) | 0.11 % |
| sslv07a | ETIREMENT D'UN PARALLELEPIPEDE SOUS SON PROPRE POIDS | 3D | CALC_ERREUR, MACRO_ELAS_MULT, MECA_STATI | 17 (0) | 0.0011 % |
| sslv07b | ETIREMENT D'UN PARALLELIPIPEDE SOUS SON PROPRE POIDS | 3D | CALC_ERREUR, MECA_STATIQUE | 13 (13) | 2.2 % |
| sslv07c | ETIREMENT D'UN PARALLELIPIPEDE SOUS SON PROPRE POIDS | 3D | MECA_STATIQUE | 13 (13) | 2.2 % |
| sslv07d | ETIREMENT D'UN PARALLELEPIPEDE SOUS SON PROPRE POIDS | 3D_INCO_UPG | CALC_ERREUR, STAT_NON_LINE | 14 (0) | 0.0019 % |
| sslv104a | comparaison SIEQ_ELGA vs VMIS(SIGM_ELGA) | 3D |  | 3 (3) | 0.044 % |
| sslv105a | PRE_IDEAS() | 3D |  | 3 (3) | 0.012 % |
| sslv111g | MODELISATION 1 : ELEMENTS MECANIQUE 3D PENTA6 | 3D | CALC_ERREUR, MECA_STATIQUE | 17 (17) | - |
| sslv114a | LES NOEUDS NE PORTENT PAS  DE DDLS DE ROTATION | 3D, D_PLAN | MECA_STATIQUE | 13 (9) | 0 % |
| sslv114b | CERTAINS NOEUDS PORTENT DES DDLS DE ROTATIONS : | 3D, COQUE_AXIS, D_PLAN,  | MECA_STATIQUE | 13 (9) | 0 % |
| sslv115a | <CALCULEL3_47> AVEC CALC_PRECONT, ON PASSE CONSCIEMMENT DE LA LDC 'SAN | 3D, BARRE, CABLE_GAINE |  | 10 (6) | 7.9e-08 % |
| sslv115b | VALIDATION DE DEFI_CABLE_BP DANS LE CAS D UN MAILLAGE PYRAMIDE | 3D, BARRE |  | 4 (0) | 7.9e-08 % |
| sslv115c | MODELISATION C : MAILLAGE TETRAEDRIQUE | 3D, BARRE |  | 2 (0) | 2e-07 % |
| sslv117a | ~ MA = CREA_MAILLAGE(MAILLAGE=MA2, CREA_MAILLE=_F(NOM="CUBE_REG", GROU | 3D_DIL | STAT_NON_LINE | 4 (0) | 4.6e-07 % |
| sslv117b | ~ MA = CREA_MAILLAGE(MAILLAGE=MA2, CREA_MAILLE=_F(NOM="CUBE_REG", GROU | 3D_DIL | STAT_NON_LINE | 4 (0) | 4.6e-07 % |
| sslv117c | ~ MA = CREA_MAILLAGE(MAILLAGE=MA2, CREA_MAILLE=_F(NOM="CUBE_REG", GROU | 3D_DIL | STAT_NON_LINE | 4 (0) | 4.6e-07 % |
| sslv117d | ~ MA = CREA_MAILLAGE(MAILLAGE=MA2, CREA_MAILLE=_F(NOM="CUBE_REG", GROU | 3D_DIL | STAT_NON_LINE | 4 (0) | 4.6e-07 % |
| sslv120a | --- STRUCTURE SUSPENDUE //Z EN A + PAS DE ROTATION | 3D | MECA_STATIQUE, STAT_NON_LINE | 54 (54) | 0.0017 % |
| sslv121a | --- STRUCTURE SUSPENDUE //Z EN A + PAS DE ROTATION | 3D | MECA_STATIQUE | 18 (18) | 0.0015 % |
| sslv130a | CYLINDRE CREUX EN INCOMPRESSIBLE (NU=0.4999) | 3D_INCO_UP, 3D_INCO_UPG, | STAT_NON_LINE | 121 (121) | 0.1 % |
| sslv130e | CYLINDRE CREUX EN INCOMPRESSIBLE (NU=0.4999) | 3D_INCO_UPG, 3D_INCO_UPO | STAT_NON_LINE | 121 (121) | 0.17 % |
| sslv130f | CYLINDRE CREUX EN INCOMPRESSIBLE (NU=0.4999) | 3D_INCO_UP, 3D_INCO_UPG, | STAT_NON_LINE | 121 (121) | 0.1 % |
| sslv131a | MATERIAU ISOTROPE TRANSVERSE  = | 3D | MECA_STATIQUE, STAT_NON_LINE | 45 (45) | 3.7e-08 % |
| sslv135a | Cas test sslv135a : cube 3D | 3D | MECA_STATIQUE, STAT_NON_LINE | 36 (0) | 0 % |
| sslv135g | CALCULATE ELASTIC FIELD | 3D | MECA_STATIQUE, STAT_NON_LINE | 10 (0) | 1.1 % |
| sslv135h | Cas test sslv135h : cube 3D | 3D | STAT_NON_LINE | 2 (0) | - |
| sslv135i | Cas test sslv135f : cube 3D | 3D | STAT_NON_LINE | 4 (0) | 1.8e-06 % |
| sslv140a | _F(GROUP_NO='FACEXY1', | 3D | MECA_STATIQUE | 4 (0) | 1.1 % |
| sslv146a | VALIDATION DES GRILLES LINEAIRE QUADRANGLE | 3D, GRILLE_MEMBRANE | MECA_STATIQUE, STAT_NON_LINE | 37 (36) | 0.0001 % |
| sslv146b | VALIDATION DES GRILLES LINEAIRE TRIANGLE | 3D, GRILLE_MEMBRANE | MECA_STATIQUE, STAT_NON_LINE | 72 (72) | 0.0001 % |
| sslv146c | VALIDATION DES GRILLES QUADRATIQUE QUADRANGLE | 3D, GRILLE_MEMBRANE | MECA_STATIQUE, STAT_NON_LINE | 36 (36) | 0.0022 % |
| sslv146d | VALIDATION DES GRILLES QUADRATIQUE TRIANGLE | 3D, GRILLE_MEMBRANE | MECA_STATIQUE, STAT_NON_LINE | 72 (72) | 0.0014 % |
| sslv157a |  | 3D, DIS_TR | MECA_STATIQUE | 12 (12) | 2.7e-05 % |
| sslv157b |  | 3D, DIS_TR | MECA_STATIQUE | 12 (12) | 2.7e-05 % |
| sslv224b | FICHIER DE COMMANDE DU TP FORMATION ASTER AVANCE | 3D, 3D_INTERFACE | STAT_NON_LINE | 1 (0) | 3 % |
| sslv303c | (3D ET FORCE_INTERNE) | 3D | MACRO_ELAS_MULT | 14 (8) | 0.059 % |
| sslv306a | DEFINITION DES GROUPES DE NOEUDS | 3D | MECA_STATIQUE | 12 (0) | 0.0001 % |
| sslv306b |  | 3D | MECA_STATIQUE | 2 (0) | - |
| sslv307a | DEFINITION DES GROUPES DE NOEUDS | 3D | MECA_STATIQUE | 56 (45) | 0.0021 % |
| sslv316l | ON IGNORE L'ALARME XFEM2_94 PARCE QUE MEME SI DANS LE CAS OU ELLE EST | 3D |  | 4 (0) | - |
| sslv320a | XFEM_69 EST IGNORE PARCE QUE CETTE ALARME NE S'APPLIQUE PAS A CE CAS T | 3D |  | 4 (0) | 0 % |
| sslv320b | XFEM_69 EST IGNORE PARCE QUE CETTE ALARME NE S'APPLIQUE PAS A CE CAS T | 3D |  | 4 (0) | 9e-06 % |
| sslv320c | METHODE FMM | 3D |  | 4 (0) | 0 % |
| sslv320d | ON IGNORE L'ALARM HOMARD0_9 CAR ON SAIT BIEN CE QU'ON FAIT ;-) | 3D |  | 4 (0) | 0 % |
| sslv321a | sslv321a.para=tps_job 30 mem_job 256Mo mem_aster 50 ncpus 1 liste_test | 3D | MECA_STATIQUE | 6 (0) | - |
| sslv321b | sslv321b.para=tps_job 30 mem_job 256Mo mem_aster 50 ncpus 1 liste_test | 3D | MECA_STATIQUE | 6 (0) | - |
| sslv321c | sslv321b.para=tps_job 30 mem_job 256Mo mem_aster 50 ncpus 1 liste_test | 3D | MECA_STATIQUE | 6 (0) | - |
| sslv321d | sslv321d.para=tps_job 30 mem_job 256Mo mem_aster 50 ncpus 1 liste_test | 3D | MECA_STATIQUE | 6 (0) | - |
| sslv321e | sslv321a.para=tps_job 30 mem_job 256Mo mem_aster 50 ncpus 1 liste_test | 3D | MECA_STATIQUE | 6 (0) | - |
