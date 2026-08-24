# Proiettore Irrotazionale — Teoria e Metodo

Questo documento raccoglie **tutti** i passaggi teorici e numerici del metodo del "proiettore
irrotazionale" (least-squares + Poisson via FFT/DCT) che sperimentiamo in questa cartella, come
alternativa veloce all'annichilazione dei vortici del paper.

Struttura: una parte principale (§1–§6) che segue il filo logico dal problema fisico
(cos'è un vortice, Helmholtz, metodo del paper, metodo nostro, come si risolve Poisson, dove si
colloca nell'algoritmo) fino alle tre righe di codice, e una serie di **appendici** (A–N) con tutte
le derivazioni di dettaglio (calcolo delle variazioni, aggiunto/trasposto, Laplaciano convolutivo,
autovalori/autovettori per FFT e DCT, condizioni al contorno, dualità di Helmholtz).

Codice di riferimento: la funzione [`irrotational_phase`](vah_projector_fft_test.py) negli script di
questa cartella. Le funzioni condivise di rilevamento/eliminazione vortici stanno nella cartella
sorella: [function_vortex_detection_accegpu.py](../1_Alternative_projection/function_vortex_detection_accegpu.py)
e [function_vortex_elimination_accegpu.py](../1_Alternative_projection/function_vortex_elimination_accegpu.py).

---

## Lavori aperti (TODO)

- [x] **Condizioni al contorno: passare da periodiche (FFT) a Neumann (DCT).** Le BC
  corrette secondo il variazionale (Appendice A/G) sono **Neumann inomogenee**
  $\partial\psi/\partial n=\mathbf g\cdot\hat{\mathbf n}$, non periodiche. Avevamo scelto le
  periodiche pensando fossero l'unica via per usare la FFT e guadagnare in velocità, ma la
  **DCT-II** diagonalizza il Laplaciano con Neumann a **costo identico** ($O(N\log N)$: è una
  FFT su segnale simmetrizzato) ed **elimina il padding di guardia** (512→768) che ci serviva
  solo per simulare Neumann sul toro. **FATTO:** solver DCT/Neumann implementato e confrontato con
  FFT in [dct_vs_fft.py](dct_vs_fft.py); teoria in
  [§5](#5-come-si-risolve-poisson-analitico-e-numerico-fft-e-dct) (DCT), [Appendice F](#appendice-f--autovaloriautovettori-con-le-esponenziali-immaginarie)
  e [Appendice G](#appendice-g--condizioni-al-contorno-neumann-periodiche-padding-solvibilità).
  **Ancora aperto:** (a) trattare il dato di Neumann **inomogeneo** $\mathbf g\cdot\hat{\mathbf n}$ sul
  bordo (ora imponiamo Neumann omogeneo); (b) **FATTO:** confronto empirico periodico↔Neumann
  consolidato su più target, con lo script parametrizzato in dimensione ologramma/SR
  ([dct_vs_fft.py](dct_vs_fft.py)).
- [x] **Nota consolidata: alzare il floor batte il masking.** Su target dark-heavy,
  regolarizzare alzando `target_floor_rel` è più efficace del weighting/masking del
  proiettore (vedi Appendice I/L). Gli script relativi (weighted/masked, alpha schedulato,
  floor sweep) sono ora **relegati in [`old/`](old/)**: danno risultati poco rilevanti su
  immagini "quadrate" (senza grandi zone nere).

---

## Riordino della cartella (24 ago 2026)

Pulizia di fine giornata. La cartella ora contiene **solo** gli script vivi; tutto ciò che era
diagnostico/accantonato è in [`old/`](old/), e le cartelle `output_*` sono state rimosse (gli
script **non salvano più** su disco: mostrano le figure con un unico `plt.show()` finale e
scrivono una tabellina `results_*.md` — entrambi gitignorati).

**Parametrizzazione comune.** Tutti gli script vivi espongono in testa la geometria in due numeri:
`HOLOGRAM_SIZE` (lato dell'apertura SLM = metà griglia, fissa il FOV totale) e `SR_FRACTION`
(lato della signal region come frazione della griglia). Tutto il resto (`WORK_SIZE`, `SR_SIZE`,
padding, offset apertura) è derivato. I target sono selezionabili commentando/scommentando una riga
e vivono in [`targets/`](targets/): `marmo`, `object_grayscale_from_mat`, `Lenna`, `Baboon`,
`valentini`.

**Script vivi:**
- [comparison_paper_vs_projector.py](comparison_paper_vs_projector.py) — **confronto definitivo** a 5
  metodi (tempo + qualità): paper periodico, nostro periodico, nostro a ogni iterazione, nostro a
  ogni iterazione con init quadratica di Chen, nostro a ogni iterazione con random filtrata ottimale.
- [dct_vs_fft.py](dct_vs_fft.py) — **DCT vs FFT** (Neumann vs periodiche).
- [vah_projector_smooth_init.py](vah_projector_smooth_init.py) — **inizializzazioni**: smooth-filtrata
  (più seed) vs random (più seed) vs quadratica di Chen (pulita, con `QUAD_C_MODE`/`QUAD_C_MANUAL`).
- [vah_projector_fft_test.py](vah_projector_fft_test.py) — **sweep del blend alpha** (0…1) vs paper.
- [vah_projector_periodic_test.py](vah_projector_periodic_test.py) — proiettore **periodico vs a ogni
  iterazione** + tempi (per-evento ~230× più veloce dell'arctan2 del paper).
- Didattici: [temp_smooth_init.py](temp_smooth_init.py),
  [temp_smooth_init_method_explaining.py](temp_smooth_init_method_explaining.py).

**Relegati in [`old/`](old/):** weighted/masked solve, alpha schedulato/strategy, weighted feedback,
floor sweep, quadratic/corrlen/smooth sweep, iterated projection, blend keep-amplitude, quality-weighted
solve, residual diagnosis, vortexfree init, e il test **on-SLM** (fallito: la fase in piano SLM è
speckle intrinseca, cfr. Appendice H). Danno segnali poco rilevanti su immagini quadrate; il loro
insegnamento resta nelle appendici I/J/L/H.

**Confronto definitivo (esempio marmo, 384/768/512, 500 iter):** paper RMSE ~11.8 (787 vortici) →
nostro-periodico ~8.9 → nostro-ogni-iter ~7.1 (0 vortici) → +Chen ~6.3 → +random-filtrata ~3.3, con
il proiettore ~230× più veloce per evento. Gli stessi risultati, runnabili e self-contained, sono
replicati in `Vortex_patching/marco_risultati`.

---

## Indice

- [1. Cos'è un vortice ottico (formalismo del paper)](#1-cosè-un-vortice-ottico-formalismo-del-paper)
- [2. Come si rilevano i vortici: paper vs nostro metodo](#2-come-si-rilevano-i-vortici-paper-vs-nostro-metodo)
- [3. Formalizzazione: il problema di minimo (Lagrange/least-squares)](#3-formalizzazione-il-problema-di-minimo-lagrangeleast-squares)
- [4. Dimostrazione: dal minimo all'equazione di Poisson](#4-dimostrazione-dal-minimo-allequazione-di-poisson)
- [5. Come si risolve Poisson: analitico e numerico (FFT e DCT)](#5-come-si-risolve-poisson-analitico-e-numerico-fft-e-dct)
- [6. Dove si colloca nell'algoritmo iterativo (GS)](#6-dove-si-colloca-nellalgoritmo-iterativo-gs)
- [Appendice A — Minimizzazione di un funzionale (1D e 2D)](#appendice-a--minimizzazione-di-un-funzionale-1d-e-2d)
- [Appendice B — Dal continuo al numerico: vettorizzazione, stencil, $A^\top A$](#appendice-b--dal-continuo-al-numerico-vettorizzazione-stencil-ata)
- [Appendice C — Aggiunto vs trasposto](#appendice-c--aggiunto-vs-trasposto)
- [Appendice D — Perché divergenza = − gradiente trasposto](#appendice-d--perché-divergenza---gradiente-trasposto)
- [Appendice E — Il Laplaciano come operatore convolutivo](#appendice-e--il-laplaciano-come-operatore-convolutivo)
- [Appendice F — Autovalori/autovettori con le esponenziali immaginarie](#appendice-f--autovaloriautovettori-con-le-esponenziali-immaginarie)
- [Appendice G — Condizioni al contorno: Neumann, periodiche, padding, solvibilità](#appendice-g--condizioni-al-contorno-neumann-periodiche-padding-solvibilità)
- [Appendice H — Nota sperimentale: conteggio vortici vs momento angolare (OAM)](#appendice-h--nota-sperimentale-conteggio-vortici-vs-momento-angolare-oam)
- [Appendice I — Nota sperimentale: sweep del floor su target dark-heavy](#appendice-i--nota-sperimentale-sweep-del-floor-su-target-dark-heavy)
- [Appendice J — Nota sperimentale: strategia di alpha (costante vs schedulato)](#appendice-j--nota-sperimentale-strategia-di-alpha-costante-vs-schedulato)
- [Appendice K — Nota sperimentale: Poisson periodico/a-stagnazione e tempi](#appendice-k--nota-sperimentale-poisson-periodicoa-stagnazione-e-tempi)
- [Appendice L — Nota sperimentale: weighted/masked Poisson solve (accantonato)](#appendice-l--nota-sperimentale-weightedmasked-poisson-solve-accantonato)
- [Appendice M — Inizializzazione: diffusore random “matched” alla signal region](#appendice-m--inizializzazione-diffusore-random-matched-alla-signal-region)
- [Appendice N — Dualità di Helmholtz: la parte rotazionale del paper](#appendice-n--dualità-di-helmholtz-la-parte-rotazionale-del-paper)

---

## 1. Cos'è un vortice ottico (formalismo del paper)

Un campo ottico è $h(\mathbf r)=a(\mathbf r)\,e^{i\varphi(\mathbf r)}$. Un **vortice ottico**
(singolarità di fase) è un punto dove l'ampiezza si annulla, $a=0$, e la fase è indefinita:
girando attorno al punto, la fase avvolge di $\pm 2\pi$.

### La catena logica: dal difetto alla circuitazione

Il ragionamento che porta all'oggetto che poi useremo è una catena precisa:

1. **Singolarità → fase indeterminata.** Dove $a=0$ il campo complesso è $0$: l'angolo
   $\varphi=\arg h$ non è definito (l'origine del piano complesso non ha argomento).
2. **Fase indeterminata → rotore del gradiente non nullo.** Attorno a quel punto la fase non è più a
   valore singolo: $\nabla\varphi$ **non** è un gradiente "vero". Il suo **rotore** non è zero, ma una
   delta concentrata nel core:
   $$\nabla\times\nabla\varphi=2\pi\sum_k q_k\,\delta(\mathbf r-\mathbf r_k).$$
   (Per una funzione liscia sarebbe identicamente $0$: $\nabla\times\nabla f\equiv 0$.)
3. **Rotore ≠ 0 → integrale di superficie ≠ 0.** Integrando il rotore su una superficie $S$ che
   racchiude il core, per il **teorema di Stokes**:
   $$\iint_S(\nabla\times\nabla\varphi)\cdot d\mathbf S=\oint_{\partial S}\nabla\varphi\cdot d\boldsymbol\ell.$$
4. **→ circuitazione ≠ 0.** Il membro di destra è la **circuitazione** della fase lungo un cammino
   chiuso attorno al vortice, e vale $2\pi q$. È **questa** la quantità che rileviamo e usiamo (la
   plaquette a 4 pixel dell'[§2](#2-come-si-rilevano-i-vortici-paper-vs-nostro-metodo) è la sua
   versione discreta).

**La circuitazione è quantizzata — e cosa rappresenta.** Perché $e^{i\varphi}$ sia a valore singolo
(torni su sé stesso dopo un giro), l'avvolgimento della fase deve essere un multiplo intero di
$2\pi$. Quindi la **carica topologica**
$$Q=\frac{1}{2\pi}\oint_C \nabla\varphi\cdot d\boldsymbol\ell \in\{\dots,-1,0,+1,\dots\}$$
è **intera**: conta quante volte la fase avvolge di $2\pi$ attorno al core (il "numero di giri"). Essendo
intera **non può cambiare con continuità**: è questo che rende il vortice un **difetto topologico** —
non lo si "smussa" con una piccola perturbazione liscia. Per rimuoverlo serve o farlo **uscire** dal
dominio, o farlo **annichilire** con un vortice di carica opposta (facendo passare l'ampiezza per
zero). Il segno di $Q$ è il verso di avvolgimento (destrorso/sinistrorso); il conteggio $N_++N_-$ e le
cariche relative sono indipendenti dalla convenzione di orientazione del cammino.

### Il vortice È uno zero di intensità

Vale anche il viceversa, e spiega perché in olografia i vortici sono **inevitabili**: un campo **a
banda limitata** (apertura finita dell'SLM) forzato ad approssimare un target reale sviluppa generici
**zeri** del campo complesso, e **ogni zero isolato di un campo complesso 2D è un vortice** (la fase
deve girare attorno a un punto in cui il campo si annulla). Ampiezza nulla e avvolgimento di fase sono
due facce dello stesso oggetto: l'intensità $I=a^2$ ha un **buco** ($I=0$) esattamente nel core. Ecco
perché i vortici, nella ricostruzione, si vedono come **punti neri** puntiformi che bucano l'immagine e
ne bloccano la convergenza (RMSE in plateau).

### Come isolarli: decomposizione di Helmholtz (irrotazionale + solenoidale)

In presenza di vortici $\nabla\varphi$ non è un gradiente puro. Il **teorema di Helmholtz** dice che
ogni campo vettoriale si decompone in una parte **irrotazionale** (gradiente di uno scalare) e una
**solenoidale/rotazionale** (rotore di un potenziale vettore):

$$\nabla\varphi=\underbrace{\nabla\psi}_{\text{irrotazionale (rotore }=0)}+\underbrace{\nabla\times\mathbf A}_{\text{solenoidale/rotazionale (divergenza }=0)}.$$

- la parte **irrotazionale** $\nabla\psi$ ha $\nabla\times\nabla\psi=0$: **nessun vortice**, è un vero
  potenziale scalare a valore singolo;
- la parte **solenoidale** $\nabla\times\mathbf A$ porta **tutta** la carica:
  $\nabla\times(\nabla\varphi)=2\pi\sum_k q_k\,\delta(\mathbf r-\mathbf r_k)$; è lì che vivono i vortici.

In 2D la parte solenoidale si scrive con un **solo scalare** (la *stream function* $\chi$), e
$\nabla\times\mathbf A$ diventa il **gradiente ruotato di 90°** $\nabla^\perp\chi=(-\partial_y\chi,\partial_x\chi)$;
la riduzione (perché in 2D rotore $=\nabla^\perp$, e perché il suo rotore è il Laplaciano) è in
[Appendice N](#appendice-n--dualità-di-helmholtz-la-parte-rotazionale-del-paper).

### Obiettivo di entrambi i metodi — e come "non fare buchi"

Vogliamo la fase **priva di vortici** $\psi$ (il potenziale scalare). Due strade complementari:
- il **paper** costruisce esplicitamente la parte **rotazionale** $\varphi_{\text{vort}}$ (i vortici) e
  la **sottrae**: $\psi=\varphi-\varphi_{\text{vort}}$;
- **noi** proiettiamo direttamente sulla parte **irrotazionale** con un least-squares.

Il punto "senza buchi" è cruciale: **non** rimuoviamo i pixel del core né tagliamo l'immagine. La parte
irrotazionale è, per costruzione, definita e liscia **ovunque** (è un vero potenziale); lontano dai
core coincide con $\varphi$ (si conserva il dettaglio dell'immagine), e solo **sul** core se ne
discosta di quel tanto che serve a disfare l'avvolgimento. Nel blend di campo ([§6](#6-dove-si-colloca-nellalgoritmo-iterativo-gs))
l'annullamento avviene tramite un **nullo di ampiezza** fisico, non cancellando dati: il buco del
vortice viene richiuso, non lasciato aperto ([Appendice N.8](#appendice-n--dualità-di-helmholtz-la-parte-rotazionale-del-paper)).

---

## 2. Come si rilevano i vortici: paper vs nostro metodo

**Rilevamento (comune a entrambi).** Si discretizza $Q$ su un anello di 4 pixel (plaquette). Con
il gradiente *wrapped* $W(t)=\mathrm{mod}(t+\pi,2\pi)-\pi$:

$$g_x=W(\Delta_x\varphi),\quad g_y=W(\Delta_y\varphi),\qquad
\text{curl}[i,j]=g_x[i,j]+g_y[i,j{+}1]-g_x[i{+}1,j]-g_y[i,j].$$

Se $\text{curl}\approx+2\pi$ → vortice positivo; se $\approx-2\pi$ → negativo (soglia `2π − 0.1`).
Questo è il cuore di [function_vortex_detection_accegpu.py](../1_Alternative_projection/function_vortex_detection_accegpu.py).

**Metodo del paper (arctan2 / funzione di Green).** Note posizioni $\mathbf r_k$ e cariche $q_k$ (dal
rilevamento), il paper ricostruisce **esplicitamente la parte rotazionale** e la sottrae. La "vite"
canonica di un vortice unitario in $\mathbf r_k$ è $\theta_k=\operatorname{atan2}(y-y_k,x-x_k)$;
sommandole pesate per la carica si ottiene una fase **vorticosa e continua**

$$\varphi_{\text{vort}}(\mathbf r)=\sum_k q_k\,\theta_k=\sum_k q_k\arg(z-z_k)=\operatorname{Im}\,\log\!\prod_k(z-z_k)^{q_k},
\qquad \varphi_{\text{free}}=(\varphi-\varphi_{\text{vort}})\bmod 2\pi.$$

Non è una costruzione euristica: $\varphi_{\text{vort}}$ è **esattamente** la parte rotazionale della
decomposizione di Helmholtz, ottenuta risolvendo $\nabla^2\chi=\text{(vorticità)}$ con la **funzione di
Green** $\tfrac1{2\pi}\ln r$ del Laplaciano. La derivazione passo-passo (Green, la produttoria dei
$(z-z_k)$, e la dimostrazione che $\nabla^\perp\chi=\nabla\varphi_{\text{vort}}$) è in
[Appendice N](#appendice-n--dualità-di-helmholtz-la-parte-rotazionale-del-paper).
Costo $O(K\,N^2)$ (un `arctan2` per vortice su tutta la griglia): esatto e fisicamente trasparente, ma
dipende dalla soglia di rilevamento e diventa lento quando i vortici $K$ sono molti.

**Il nostro metodo (least-squares → Poisson via FFT/DCT).** Non localizziamo i vortici. Calcoliamo
direttamente la parte irrotazionale come **proiezione ai minimi quadrati** del gradiente misurato,
risolvendo un'equazione di Poisson con **una singola trasformata** (FFT periodica o DCT/Neumann,
[§5](#5-come-si-risolve-poisson-analitico-e-numerico-fft-e-dct)): costo $O(N\log N)$, **indipendente
dal numero di vortici**. È la funzione [`irrotational_phase`](vah_projector_fft_test.py).

---

## 3. Formalizzazione: il problema di minimo (Lagrange/least-squares)

Cerchiamo il campo scalare $\psi$ il cui gradiente approssima al meglio il gradiente *wrapped*
misurato $\mathbf g=W(\nabla\varphi)$:

$$\psi=\arg\min_\psi \int_\Omega\|\nabla\psi-\mathbf g\|^2\,dA
=\arg\min_\psi \int_\Omega\big[(\psi_x-g_x)^2+(\psi_y-g_y)^2\big]\,dA.$$

Per costruzione la soluzione $\psi$ è un vero gradiente ($\nabla\times\nabla\psi=0$): **zero carica
ovunque, nessun vortice**. È la **proiezione di Hodge discreta**: tiene la componente integrabile e
scarta quella rotazionale. La dimostrazione formale (calcolo delle variazioni) è in
[Appendice A](#appendice-a--minimizzazione-di-un-funzionale-1d-e-2d).

---

## 4. Dimostrazione: dal minimo all'equazione di Poisson

Annullando la derivata funzionale del funzionale $J[\psi]$ (equazioni di Eulero–Lagrange = equazioni
normali del least-squares) si ottiene l'**equazione di Poisson**

$$\boxed{\;\nabla^2\psi=\nabla\cdot\mathbf g\;}\qquad \rho\equiv\nabla\cdot\mathbf g=\nabla\cdot W(\nabla\varphi),$$

con condizione al contorno **naturale** di Neumann inomogenea $\partial\psi/\partial n=\mathbf g\cdot\hat{\mathbf n}$
sul bordo. Derivazione completa 1D e 2D in [Appendice A](#appendice-a--minimizzazione-di-un-funzionale-1d-e-2d);
la struttura discreta $A^\top A\,\psi=A^\top\mathbf g$ (equazioni normali = Poisson discreto) è in
[Appendice B](#appendice-b--dal-continuo-al-numerico-vettorizzazione-stencil-ata).

La **condizione di solvibilità** ($\int_\Omega\rho=\oint_{\partial\Omega}\mathbf g\cdot\hat{\mathbf n}$) è
automaticamente soddisfatta perché $\rho=\nabla\cdot\mathbf g$ (teorema della divergenza); vedi
[Appendice G](#appendice-g--condizioni-al-contorno-neumann-periodiche-padding-solvibilità).

---

## 5. Come si risolve Poisson: analitico e numerico (FFT e DCT)

Abbiamo il problema (§3–§4): $\nabla^2\psi=\rho=\nabla\cdot\mathbf g$ con la condizione al contorno
**naturale di Neumann** $\partial\psi/\partial n=\mathbf g\cdot\hat{\mathbf n}$. Qui lo risolviamo,
seguendo il filo **dal continuo al codice**.

### 5.1 — Analitico: il Laplaciano è diagonale in Fourier

Le esponenziali sono autofunzioni del Laplaciano:
$$\nabla^2 e^{i\mathbf k\cdot\mathbf r}=-|\mathbf k|^2\,e^{i\mathbf k\cdot\mathbf r}
\quad\Longrightarrow\quad \hat\psi(\mathbf k)=\frac{\hat\rho(\mathbf k)}{-|\mathbf k|^2}.$$
Cioè **trasformo, divido per $-|\mathbf k|^2$, antitrasformo**. Analiticamente è tutto qui.

> **Parentesi — condizione di solvibilità.** Il modo $\mathbf k=0$ (la costante) ha autovalore $0$: non
> si può dividere. È la **costante libera** del problema di Neumann e impone la **compatibilità**
> $\int_\Omega\rho=\oint_{\partial\Omega}\mathbf g\cdot\hat{\mathbf n}$ (si ottiene integrando la PDE su
> $\Omega$ e usando il teorema della divergenza; nasce dall'alternativa di Fredholm, perché la costante
> è nel nucleo). Da noi è **automatica**: $\rho=\nabla\cdot\mathbf g$ la rende il teorema della
> divergenza applicato a $\mathbf g$, cioè un'**identità**, non un vincolo da verificare. La costante
> residua la fissiamo con la gauge a **media nulla** (§5.4). Derivazione completa:
> [Appendice G](#appendice-g--condizioni-al-contorno-neumann-periodiche-padding-solvibilità).

### 5.2 — Numericamente serve il Laplaciano DISCRETO: è una convoluzione

Sul computer non applichiamo il $\nabla^2$ continuo: usiamo differenze finite. Serve un teorema.

**Teorema (LTI = convoluzione).** Ogni operatore **lineare e invariante per traslazione** $L$ è una
**convoluzione**, con kernel = **risposta all'impulso** $k=L\delta$. *Dimostrazione* (identica in
continuo e discreto): ogni funzione si scrive come sua convoluzione con la delta (proprietà di
**setaccio**) $f=\int f(x')\,\delta(\cdot-x')\,dx'$; per **linearità** $L$ entra nell'integrale e agisce
solo sulla delta (i pesi $f(x')$ sono costanti); per **invarianza per traslazione**
$L\,\delta(\cdot-x')=k(\cdot-x')$; quindi $Lf=\int f(x')\,k(\cdot-x')\,dx'=f*k$. (Carattere convolutivo
del Laplaciano: [Appendice E](#appendice-e--il-laplaciano-come-operatore-convolutivo).)

**Troviamo il kernel (caso numerico).** Applico la definizione a differenze finite:
- fase $\psi$;
- **derivata prima** (forward): $\Delta^+\psi_j=\psi_{j+1}-\psi_j$;
- **derivata seconda** (backward∘forward, per restare centrata — [Appendice B](#appendice-b--dal-continuo-al-numerico-vettorizzazione-stencil-ata)):
  $\Delta^-\Delta^+\psi_j=\psi_{j+1}-2\psi_j+\psi_{j-1}$ → **stencil a 3 punti** $[1,-2,1]$;
- in **2D** sommo le due direzioni $\partial^2_x+\partial^2_y$ → **stencil a 5 punti**
  $\left(\begin{smallmatrix}&1&\\1&-4&1\\&1&\end{smallmatrix}\right)$.

Questo stencil **è** il kernel di convoluzione del Laplaciano discreto (= $\nabla^2\delta$ campionato).

### 5.3 — Diagonalizzare la convoluzione: gli autovettori dello shift sono Fourier

**La base di Fourier — com'è costruita e perché wrap-around.** $v_j=e^{i2\pi k j/N}$. La quantizzazione
la impone la **periodicità (wrap-around)** $v_{j+N}=v_j$: $e^{ik'N}=1\Rightarrow k'=2\pi k/N$. Cioè la
base esponenziale **assume che il dominio si richiuda su sé stesso** (anello/toro).

**Autovettori dello shift.** Lo shift $S$ ($(S\psi)_j=\psi_{j+1}$) agisce sui modi come **fase pura**:
$Sv=\omega v$, con $\omega=e^{i2\pi k/N}$. I modi di Fourier sono gli autovettori **comuni** di tutti
gli shift.

**Teorema (convoluzione = combinazione lineare di shift).** Ogni convoluzione è
$k*=\sum_a k_a\,S^a$ (somma pesata di shift, pesi = kernel). Quindi ha $v$ come autovettore, con
autovalore = **la stessa combinazione valutata in $\omega$** = **FFT del kernel**. (Dettaglio,
tabella degli operatori di differenza ed esempi numerici: [Appendice F](#appendice-f--autovaloriautovettori-con-le-esponenziali-immaginarie).)

- **1D:** $L=S-2I+S^{-1}$ → $\lambda_k=\omega-2+\omega^{-1}=2\cos\frac{2\pi k}{N}-2$.
- **2D:** sommo le direzioni → $\lambda_{k,l}=2\cos\frac{2\pi k}{n}+2\cos\frac{2\pi l}{m}-4$
  ($\approx-|\mathbf k|^2$ a bassa frequenza; deviazione a Nyquist in [Appendice E](#appendice-e--il-laplaciano-come-operatore-convolutivo)/[F](#appendice-f--autovaloriautovettori-con-le-esponenziali-immaginarie)).

A questo punto **so risolvere il problema** con il proiettore FFT.

### 5.4 — Il solve FFT e le due accortezze numeriche

Le tre righe centrali ($\hat\psi_{k,l}=\hat\rho_{k,l}/\lambda_{k,l}$) di
[`irrotational_phase`](vah_projector_fft_test.py):

```python
_denom = 2*cp.cos(2*cp.pi*_ii/n) + 2*cp.cos(2*cp.pi*_jj/m) - 4
_denom[0, 0] = 1.0                     # evita 0/0 sul modo DC

rho_hat = cp.fft.fft2(rho)             # F{rho}
rho_hat[0, 0] = 0.0                    # media nulla / solvibilità
psi = cp.real(cp.fft.ifft2(rho_hat / _denom))
```

**Solo due accortezze numeriche:**
1. **La soluzione è definita a meno di una costante** — il modo DC ha $\lambda_{0,0}=0$ (aggiungere una
   costante a $\psi$ non cambia $\nabla^2\psi$). La **fissiamo** con `rho_hat[0,0]=0`: sceglie la
   soluzione a **media nulla** e insieme impone la solvibilità (§5.1). `_denom[0,0]=1` serve solo a non
   fare $0/0$ (il numeratore è già zero).
2. **`cp.real`** — $\psi$ è reale (per simmetria hermitiana, [Appendice F](#appendice-f--autovaloriautovettori-con-le-esponenziali-immaginarie)); scartiamo il residuo
   immaginario numerico ($\sim10^{-16}$).

**Padding — dove/perché/quanto (nota onesta).** Nel codice la riga
`F = np.pad(F, ((n//4,n//4),(m//4,m//4)))` (512→768) è il **band-limit dell'SLM** (oversampling di Chen:
griglia 768, apertura 384, SR 512), **non** un guard-band per Poisson. Il solve FFT vero
(`irrotational_phase`) gira sul **ritaglio 512** e **tollera** la piccola cucitura periodica ai bordi
(in pratica trascurabile). Il "padding di guardia per far sì che la FFT-periodica *approssimi* Neumann"
è un rimedio **teorico** ([Appendice G](#appendice-g--condizioni-al-contorno-neumann-periodiche-padding-solvibilità)),
un'opzione non applicata in questo script. La DCT (§5.5) lo rende del tutto inutile.

### 5.5 — Condizioni al contorno: il mio Laplaciano è "a specchio" → DCT

Numericamente, per fare le differenze **ai bordi** serve un'assunzione (cosa c'è oltre l'ultimo pixel).
**La mia costruzione la contiene già, ed è a specchio** (non wrap):

- `dx[:,-1]=0` (forward troncato, niente wrap) + `rho[:,0]=dx[:,0]` danno **mezzi stencil** ($-1$ agli
  angoli), **identici** allo stencil pieno col fantasma **a specchio** $\psi_{-1}=\psi_0$;
- e $\psi_{-1}=\psi_0$ significa **pendenza nulla al bordo** ($\psi_0-\psi_{-1}=0$), cioè
  $\partial\psi/\partial n=0$: è **Neumann**. (Il dato **inomogeneo** $\mathbf g\cdot\hat{\mathbf n}$ è
  nella RHS: la riga di bordo di `rho` è $g_0$. Dal minimo $\partial J/\partial\psi_0=0\Rightarrow
  \psi_1-\psi_0=g_0$, cioè $\partial\psi/\partial n=\mathbf g\cdot\hat{\mathbf n}$;
  [Appendice A](#appendice-a--minimizzazione-di-un-funzionale-1d-e-2d).)

Quindi, **a rigore, questo operatore non va diagonalizzato con gli esponenziali (periodici) ma con i
coseni (a specchio): la DCT.**

**Cosa sono i coseni (DCT-II) e la specularità.** $v_j=\cos\!\big(a(j+\tfrac12)\big)$, $a=\pi k/N$. Il
**mezzo-passo** $(j+\tfrac12)$ rende $v$ pari attorno al bordo sinistro ($v_{-1}=v_0$, automatico); la
**quantizzazione** $a=\pi k/N$ la rende pari anche a destra ($v_N=v_{N-1}$, perché
$\cos(a(N{+}\tfrac12))-\cos(a(N{-}\tfrac12))=-2\sin(aN)\sin\tfrac a2=0$ richiede $\sin(\pi k)=0$). Sono i
modi di un dominio **riflesso** (periodo effettivo $2N$).

**Ricavo gli autovalori** (stesso metodo: operatore = combinazione di shift). Il coseno **non** è
autovettore di $S$ da solo, ma **lo è di $S+S^{-1}$**:
$$\big(S+S^{-1}\big)v_j=\cos\!\big(a(j{+}\tfrac12){+}a\big)+\cos\!\big(a(j{+}\tfrac12){-}a\big)=2\cos(a)\,v_j,$$
quindi $L=(S+S^{-1})-2I$ dà $\lambda_k^{\text{Neu}}=2\cos a-2=2\cos\frac{\pi k}{N}-2$ (2D: somma delle
direzioni). Il $\pi$ invece di $2\pi$ è il **raddoppio** del dominio riflesso.

**Stesso processo, nuovo autovalore, DCT.** Cambiano solo trasformata e autovalori; `rho` è identico:

```python
_denom_dct = (2*cp.cos(cp.pi*_ii/n) - 2) + (2*cp.cos(cp.pi*_jj/m) - 2)
rho_hat = dctn(rho, type=2, norm="ortho")
phi_hat = rho_hat / _denom_dct
phi_hat[0, 0] = 0.0                        # gauge: media nulla (la costante libera, come in §5.4)
psi = idctn(phi_hat, type=2, norm="ortho")
```

Anche qui l'unica scelta libera è la **media nulla** (`phi_hat[0,0]=0`): la stessa costante di §5.4.

**FFT vs DCT — in una riga.** Stessa equazione, stesso costo $O(N\log N)$: la FFT diagonalizza il
Laplaciano **periodico** (wrap), la DCT quello di **Neumann** (specchio) — che è quello che **abbiamo
davvero costruito**. Quindi la DCT è la scelta **consistente e corretta**, e non serve alcun padding di
guardia. Confronto empirico: [dct_vs_fft.py](dct_vs_fft.py). Autovalori
in dettaglio: [Appendice F](#appendice-f--autovaloriautovettori-con-le-esponenziali-immaginarie);
condizioni al contorno: [Appendice G](#appendice-g--condizioni-al-contorno-neumann-periodiche-padding-solvibilità).

---

## 6. Dove si colloca nell'algoritmo iterativo (GS)

Il proiettore **non** è un algoritmo a sé: è un **hook** dentro il loop Gerchberg–Saxton a banda
limitata, chiamato una volta per iterazione **sul piano immagine** (mai sul piano SLM —
[Appendice H](#appendice-h--nota-sperimentale-conteggio-vortici-vs-momento-angolare-oam)). Struttura
del loop (in [`vah_projector_fft_test.py`](vah_projector_fft_test.py)):

```text
inizializza fase phi  (random, oppure init "matched" — Appendice M)
ripeti per ogni iterazione:
  1. piano immagine: imponi ampiezza  amp = sqrt(F) nella SR, libera nella NR
  2. FFT  ->  piano SLM (Fourier)
  3. piano SLM: imponi ampiezza incidente (gaussiana) dentro l'apertura (band-limit)
  4. IFFT ->  piano immagine:   es
  5. estrai la fase           pha = angle(es)                     # <-- QUI
  6. [detection] conta i vortici su pha (plaquette)               # solo diagnostica
  7. [PROIETTORE] psi = irrotational_phase(pha_SR)                # Poisson FFT o DCT
                  psi += allineamento offset globale
                  field = (1-alpha)*e^{i*pha} + alpha*e^{i*psi}   # blend di CAMPO
                  pha_SR <- angle(field)
  8. phi = e^{i*pha}  (con la SR corretta)   ->   torna al passo 1
```

Punti chiave sul **dove/come**:

- **Dove:** subito dopo il passo 4 (IFFT→immagine), sulla fase `angle(es)` **ristretta alla signal
  region** (`pha_crop`). È lì che i vortici sono difetti reali della ricostruzione.
- **Cosa sostituisce:** nel paper qui c'è l'annichilazione arctan2, applicata **periodicamente** (ogni
  $x$ iterazioni) perché è una sostituzione dura; il nostro proiettore è **soft** (blend con $\alpha$) e
  si applica **ogni** iterazione, sopprimendo i vortici man mano che si riformano.
- **Il blend è nel dominio del campo** ($(1-\alpha)e^{i\varphi}+\alpha e^{i\psi}$, non
  $\varphi+\alpha(\psi-\varphi)$): così l'ampiezza **si abbassa** naturalmente ai core, permettendo
  l'annichilazione topologica (il winding può cambiare solo passando per un nullo di ampiezza —
  [Appendice N.8](#appendice-n--dualità-di-helmholtz-la-parte-rotazionale-del-paper)). $\alpha=0$ è il
  GS puro; $\alpha=1$ è la fase irrotazionale piena; $0<\alpha<1$ interpola.
- **Allineamento dell'offset globale:** $\psi$ dal Poisson è a media nulla (gauge), mentre `pha` ha
  media arbitraria; prima del blend si riallinea l'offset (`psi += angle(sum(exp(i(pha-psi))))`),
  altrimenti il blend somma due phasori globalmente ruotati e introduce dip/shift ovunque.
- **Regolarizzazione pratica:** il floor sul target (Appendice I) definisce la fase dove l'intensità è
  $\sim 0$ e spegne il rumore alla sorgente; $\alpha$ costante $\sim 0.5$–$0.75$ (Appendice J) è la
  scelta più semplice e robusta.

---

# Appendici

## Appendice A — Minimizzazione di un funzionale (1D e 2D)

Un **funzionale** mangia una funzione e restituisce un numero. Il nostro è
$J[\psi]=\int_\Omega L\,dA$ con densità $L=(\psi_x-g_x)^2+(\psi_y-g_y)^2$ (dipende solo dalle derivate).

**Idea variazionale.** Si perturba $\psi\to\psi+\varepsilon\eta$ con $\eta$ funzione test arbitraria,
si definisce $\Phi(\varepsilon)=J[\psi+\varepsilon\eta]$ e si impone $\Phi'(0)=0$ per ogni $\eta$.

### Derivazione 1D

Con $J[\psi]=\int_a^b(\psi_y-g_y)^2\,dy$:

$$\Phi'(0)=2\int_a^b(\psi_y-g_y)\,\eta_y\,dy
\stackrel{\text{parti}}{=}2\big[(\psi_y-g_y)\,\eta\big]_a^b-2\int_a^b(\psi_{yy}-g_{yy})\,\eta\,dy.$$

Perché sia nullo per ogni $\eta$:
- **interno:** $\psi_{yy}=g_{yy}$ (Poisson 1D);
- **bordo:** $(\psi_y-g_y)\big|_{a,b}=0$ (Neumann inomogenea) — la *natural boundary condition* che
  emerge quando non si vincola $\psi$ al bordo. Se invece si fissa $\psi$ al bordo (Dirichlet), allora
  $\eta=0$ lì e il termine di bordo sparisce da solo.

### Derivazione 2D

Con $L=(\psi_x-g_x)^2+(\psi_y-g_y)^2$ e l'identità di Green
$\int_\Omega\mathbf F\cdot\nabla\eta\,dA=-\int_\Omega(\nabla\cdot\mathbf F)\eta\,dA+\oint_{\partial\Omega}(\mathbf F\cdot\hat{\mathbf n})\eta\,ds$
con $\mathbf F=\nabla\psi-\mathbf g$:

$$\Phi'(0)=-2\int_\Omega\big[\nabla\cdot(\nabla\psi-\mathbf g)\big]\eta\,dA+2\oint_{\partial\Omega}(\nabla\psi-\mathbf g)\cdot\hat{\mathbf n}\,\eta\,ds.$$

Da cui, per ogni $\eta$:

$$\boxed{\nabla^2\psi=\nabla\cdot\mathbf g}\ \text{(interno)},\qquad
\boxed{\partial\psi/\partial n=\mathbf g\cdot\hat{\mathbf n}}\ \text{(bordo, Neumann)}.$$

### Via Euler–Lagrange diretta

Con la formula $\dfrac{\delta J}{\delta\psi}=\dfrac{\partial L}{\partial\psi}-\partial_x\dfrac{\partial L}{\partial\psi_x}-\partial_y\dfrac{\partial L}{\partial\psi_y}=0$
e $\partial L/\partial\psi=0$, $\partial L/\partial\psi_x=2(\psi_x-g_x)$, $\partial L/\partial\psi_y=2(\psi_y-g_y)$:

$$-2(\psi_{xx}-\partial_x g_x)-2(\psi_{yy}-\partial_y g_y)=0\ \Longrightarrow\ \nabla^2\psi=\nabla\cdot\mathbf g.$$

Poisson **è** l'equazione di Eulero–Lagrange del funzionale: non due cose diverse, la stessa equazione.

---

## Appendice B — Dal continuo al numerico: vettorizzazione, stencil, $A^\top A$

### Vettorizzazione

Una griglia $\psi_{i,j}$ di dimensione $n\times m$ si "srotola" in un vettore lungo $N=n\cdot m$. Con
ordinamento row-major $p=i\,m+j$, i vicini di griglia diventano vicini a distanza fissa nel vettore:
$(i,j+1)\to p+1$ (orizzontale), $(i+1,j)\to p+m$ (verticale). La matrice $A$ del gradiente **esiste**
($N\times N$, sparsa) ma **non si costruisce mai**: si applica come convoluzione col kernel.

### Forward + backward = seconda derivata

- **forward** $\;(\Delta^+\psi)_j=\psi_{j+1}-\psi_j$ — nel codice `dx`/`dy` (ultima riga/colonna
  azzerata = contorno);
- **backward** $\;(\Delta^-u)_j=u_j-u_{j-1}$.

Componendo backward∘forward:

$$\Delta^-\Delta^+\psi=(\psi_{j+1}-\psi_j)-(\psi_j-\psi_{j-1})=\psi_{j+1}-2\psi_j+\psi_{j-1}\quad(\text{stencil 1D }[+1,-2,+1]).$$

Sommando le due direzioni si ottiene lo **stencil a 5 punti**:

$$\nabla^2=\begin{matrix} & +1 & \\ +1 & -4 & +1 \\ & +1 & \end{matrix}$$

### Struttura least-squares $A^\top A$

Minimizzare $\|A\psi-\mathbf g\|^2$ (con $A=\Delta^+$ = forward) dà le **equazioni normali**
$A^\top A\,\psi=A^\top\mathbf g$. Poiché $A^\top=-\Delta^-$ (backward, vedi [Appendice D](#appendice-d--perché-divergenza---gradiente-trasposto)):

- $A^\top A=-\Delta^-\Delta^+=$ **Laplaciano discreto** (stencil a 5 punti);
- $A^\top\mathbf g=-\Delta^-\mathbf g=$ **divergenza discreta** $=\rho$.

Corrispondenza col codice di [`irrotational_phase`](vah_projector_fft_test.py):

| Oggetto matriciale | Riga di codice | Significato |
|---|---|---|
| $A=\Delta^+$ (forward) | `dx = pha[:,1:]-pha[:,:-1]` (ultima col=0) | gradiente, contorno troncato |
| $A^\top=-\Delta^-$ (backward) | `rho[:,1:] = dx[:,1:]-dx[:,:-1]` (+ righe di bordo) | divergenza $=\rho$ |
| $A^\top A$ (5 punti) | `/ _denom` in Fourier | inversione del Laplaciano |

In 2D, con i prodotti di Kronecker $A_x=I_n\otimes D_m$, $A_y=D_n\otimes I_m$:

$$A^\top A=A_x^\top A_x+A_y^\top A_y=I_n\otimes L_{1D}+L_{1D}\otimes I_m\quad(\text{somma di Kronecker}=\text{stencil 5 punti}).$$

---

## Appendice C — Aggiunto vs trasposto

**Trasposto** $A^\top$ (prodotto scalare reale $\langle a,b\rangle=\sum_i a_ib_i$): l'unico operatore con
$\langle Au,v\rangle=\langle u,A^\top v\rangle$. Concretamente scambia righe↔colonne: $(A^\top)_{ij}=A_{ji}$.

**Aggiunto** $A^*$ (prodotto scalare complesso $\langle a,b\rangle=\sum_i\overline{a_i}b_i$): l'unico con
$\langle Au,v\rangle=\langle u,A^*v\rangle$. Concretamente trasposta **coniugata**: $A^*=\overline{A^\top}$.

Relazione: $A^*=\overline{A^\top}$; se l'operatore è **reale** i due coincidono, $A^*=A^\top$.

**Quale usiamo:** i nostri operatori di differenza hanno coefficienti $\pm1$ (reali) ⇒ **trasposto e
aggiunto coincidono**. "Aggiunto" è il concetto astratto (vale anche su spazi complessi di funzioni),
"trasposto" la sua realizzazione come matrice reale. La distinzione conta solo passando a Fourier
(esponenziali complesse), ma l'operatore differenza resta reale.

---

## Appendice D — Perché divergenza = − gradiente trasposto

Il Laplaciano **non** è "gradiente due volte": è $\operatorname{div}\circ\operatorname{grad}$, cioè
operatore + suo **aggiunto**. In 1D sembrano uguali (entrambi $\partial_y$) ma è un caso particolare;
in 2D grad (scalare→vettore) e div (vettore→scalare) sono operatori diversi.

### Analitico (integrazione per parti)

Con prodotti scalari integrali $\langle\psi,\phi\rangle=\int\psi\phi$, $\langle\mathbf F,\mathbf G\rangle=\int\mathbf F\cdot\mathbf G$:

$$\langle\operatorname{grad}\psi,\mathbf F\rangle=\int_\Omega\nabla\psi\cdot\mathbf F\,dA
=-\int_\Omega\psi\,(\nabla\cdot\mathbf F)\,dA+\oint_{\partial\Omega}\psi\,(\mathbf F\cdot\hat{\mathbf n})\,ds
=\langle\psi,-\operatorname{div}\mathbf F\rangle+\text{bordo}.$$

Dunque $\operatorname{div}=-\operatorname{grad}^*$: **l'integrazione per parti sposta la derivata
dall'altra parte del prodotto scalare**, che è la definizione stessa di aggiunto/trasposto. Il segno
meno e il termine di bordo sono i "residui" del passaggio.

### Numerico (cambio di indice nella somma)

Con $\rho_j=g_j-g_{j-1}$ (backward) e la somma per parti discreta:

$$\langle\Delta^+u,v\rangle=\sum_j(u_{j+1}-u_j)v_j=\sum_j u_j(v_{j-1}-v_j)=\langle u,-\Delta^- v\rangle
\ \Longrightarrow\ (\Delta^+)^\top=-\Delta^-.$$

La trasposta della forward-difference **è** (meno) la backward-difference: gemello discreto di
$\operatorname{div}=-\operatorname{grad}^*$. Perciò il Laplaciano ha la forma $A^\top A$ (vedi
[Appendice B](#appendice-b--dal-continuo-al-numerico-vettorizzazione-stencil-ata)).

---

## Appendice E — Il Laplaciano come operatore convolutivo

**Non è un trucco numerico:** è un fatto analitico. Ogni operatore lineare invariante per traslazione
(LTI) è una convoluzione (teorema di Schwartz), e $\nabla^2$ lo è.

### Analitico

$$\nabla^2 f=f*(\nabla^2\delta),$$

dove il kernel $\nabla^2\delta$ è una **distribuzione** (derivata seconda della delta), singolare. In
Fourier è la moltiplicazione per la funzione di trasferimento $-|\mathbf k|^2$:

$$\mathcal F\{\nabla^2 f\}=-|\mathbf k|^2\,\hat f.$$

### Numerico

Il campionamento trasforma la distribuzione singolare in un **kernel finito** (lo stencil `[+1,-2,+1]`
/ 5 punti) e $-|\mathbf k|^2$ nel suo campionato `_denom`. Sviluppo di Taylor ($2\cos\theta-2\approx-\theta^2$):

$$\lambda_{k,l}=2\cos\tfrac{2\pi k}{n}+2\cos\tfrac{2\pi l}{m}-4\approx-\Big(\tfrac{2\pi k}{n}\Big)^2-\Big(\tfrac{2\pi l}{m}\Big)^2=-|\mathbf k|^2.$$

Quindi lo stencil e `_denom` sono le versioni **campionate** del kernel distribuzionale e della funzione
di trasferimento continua. Deviano dal Laplaciano vero solo alle alte frequenze (dove il coseno "piega"
rispetto alla parabola).

---

## Appendice F — Autovalori/autovettori con le esponenziali immaginarie

### F.0 — Cosa vuol dire qui "autovalore/autovettore", e perché risolve Poisson

Un **autovettore** di un operatore lineare $L$ è un vettore $v$ che $L$ **non ruota**, si limita a
**riscalarlo**: $Lv=\lambda v$, con $\lambda$ (l'**autovalore**) il fattore di scala. Se conosciamo una
**base** fatta tutta di autovettori, l'operatore in quella base è **diagonale**: agisce su ogni
componente separatamente, moltiplicandola per il suo $\lambda$.

**Perché ci interessa.** Risolvere $\nabla^2\psi=\rho$ è, in generale, un sistema lineare enorme
($N=n\cdot m$ incognite **accoppiate**). Ma nella base giusta il Laplaciano è diagonale, e il sistema si
**disaccoppia** in $N$ equazioni scalari **indipendenti**, una per modo:
$$\lambda_k\,\hat\psi_k=\hat\rho_k\quad\Longrightarrow\quad \hat\psi_k=\frac{\hat\rho_k}{\lambda_k}.$$
Cioè: (1) trasformo $\rho$ nella base degli autovettori (è la FFT/DCT: $\hat\rho=\text{trasformata}(\rho)$),
(2) **divido** componente per componente per gli autovalori $\lambda_k$, (3) antitrasformo. La divisione
`rho_hat / _denom` nel codice **è** esattamente questo: `_denom` è il vettore/matrice degli autovalori.

**Perché il Laplaciano ha come autovettori proprio esponenziali/coseni.** Perché è **invariante per
traslazione**: applicare lo stencil $[1,-2,1]$ e poi traslare dà lo stesso risultato che traslare e poi
applicarlo. Un operatore invariante per traslazione con BC periodiche è una matrice **circolante**,
diagonalizzata dalle **esponenziali** $e^{2\pi i kj/n}$ (la FFT); con BC di Neumann la matrice è
"circolante riflessa" e gli autovettori diventano i **coseni** (la DCT). È la **BC** a decidere *quali*
esponenziali/coseni — cioè *quale* trasformata — diagonalizza.

### Continuo: autovalore $ik$

Con autofunzione $e^{ikx}$: $\partial_x e^{ikx}=ik\,e^{ikx}$ (autovalore $ik$),
$\partial_{xx}e^{ikx}=-k^2 e^{ikx}$, e in 2D $\nabla^2 e^{i\mathbf k\cdot\mathbf r}=-|\mathbf k|^2 e^{i\mathbf k\cdot\mathbf r}$.

### Discreto: shift = fase pura $\omega$

Autovettore campionato $v_j=e^{2\pi i k j/n}$. Lo **shift** $S$ agisce come fase pura:
$(Sv)_j=v_{j+1}=\omega\,v_j$ con $\omega=e^{2\pi i k/n}$. Tutti gli operatori di differenza hanno $v$
come autovettore:

| Operatore | Espressione | Autovalore su $v$ |
|---|---|---|
| forward $D^+=S-I$ | $v_{j+1}-v_j$ | $\omega-1$ |
| backward $D^-=I-S^{-1}$ | $v_j-v_{j-1}$ | $1-\omega^{-1}$ |
| Laplaciano $S-2I+S^{-1}$ | $v_{j+1}-2v_j+v_{j-1}$ | $\omega-2+\omega^{-1}$ |

Per basse frequenze $\omega-1=e^{2\pi ik/n}-1\approx i\,\tfrac{2\pi k}{n}=ik_{\text{griglia}}$: il
forward-difference riproduce l'autovalore $ik$ del continuo.

### Da dove vengono i coseni

$$\omega-2+\omega^{-1}=\underbrace{e^{2\pi ik/n}+e^{-2\pi ik/n}}_{2\cos(2\pi k/n)}-2=2\cos\tfrac{2\pi k}{n}-2\approx -k^2.$$

Il coseno appare **solo** perché si raggruppano le esponenziali coniugate $+k$ e $-k$. In esponenziali
pure l'autovalore è la forma "pulita" $\omega-2+\omega^{-1}$; il codice usa i coseni perché così
`_denom` è **reale** (comodo per dividere numeri reali).

### F.1 — Esempio numerico esplicito: $N=4$, BC periodiche (FFT)

Prendiamo il Laplaciano 1D discreto (stencil $[1,-2,1]$) su $N=4$ punti con **BC periodiche**: la
matrice è **circolante** (l'ultimo pixel si ricollega al primo):
$$L_{\text{per}}=\begin{pmatrix}-2&1&0&1\\ 1&-2&1&0\\ 0&1&-2&1\\ 1&0&1&-2\end{pmatrix}.$$
I suoi **autovettori** sono i quattro modi di Fourier $v^{(k)}_j=e^{2\pi i kj/4}=i^{\,kj}$:
$$v^{(0)}=(1,1,1,1),\ \ v^{(1)}=(1,i,-1,-i),\ \ v^{(2)}=(1,-1,1,-1),\ \ v^{(3)}=(1,-i,-1,i).$$
Gli **autovalori** sono $\lambda_k=2\cos\frac{2\pi k}{4}-2$:

| $k$ | $2\cos(2\pi k/4)-2$ | $\lambda_k$ |
|---|---|---|
| 0 | $2\cdot 1-2$ | $\mathbf{0}$ |
| 1 | $2\cdot 0-2$ | $-2$ |
| 2 | $2\cdot(-1)-2$ | $-4$ |
| 3 | $2\cdot 0-2$ | $-2$ |

**Verifica diretta** (nessuna magia): applichiamo $L_{\text{per}}$ a $v^{(1)}=(1,i,-1,-i)$. Riga 0
(stencil $[-2,1,0,1]$): $-2\cdot 1+1\cdot i+1\cdot(-i)=-2$, e $\lambda_1 v^{(1)}_0=-2\cdot 1=-2$. ✓
Riga 1 (stencil $[1,-2,1,0]$): $1\cdot 1-2\cdot i+1\cdot(-1)=-2i=\lambda_1 v^{(1)}_1$. ✓ Vale su tutte
le componenti: $v^{(1)}$ **è** autovettore con $\lambda_1=-2$.

**Osservazioni che ricorrono nel codice:**
- $\lambda_0=0$: la costante $v^{(0)}=(1,1,1,1)$ sta nel **nullo** del Laplaciano (aggiungere una
  costante a $\psi$ non cambia $\nabla^2\psi$). Per questo il modo DC è indeterminato e lo fissiamo:
  `rho_hat[0,0]=0` (media nulla), `_denom[0,0]=1` solo per non fare $0/0$.
- $|\lambda|$ **cresce con la frequenza** ($0,2,4,2$): il modo più oscillante ($k=2$, $(1,-1,1,-1)$) ha
  il $\lambda$ più grande in modulo. Dividere per $\lambda$ **smorza** le alte frequenze → $\psi$ esce
  più liscio di $\rho$ (il Laplaciano inverso è un integratore/lisciatore).
- Solve modo-per-modo: $\hat\psi_k=\hat\rho_k/\lambda_k$ per $k\neq 0$, $\hat\psi_0=0$. Tre righe.

**Nota (perché $\psi$ è reale).** $\rho$ reale ⇒ $\hat\rho$ hermitiano; $\lambda_{k,l}$ è reale e pari
($\lambda_{-k,-l}=\lambda_{k,l}$), quindi $\hat\rho/\lambda$ resta hermitiano e la sua IFFT è reale.
`cp.real(...)` scarta solo il residuo numerico ($\sim10^{-16}$) e converte il tipo complesso→float.

### La base DCT: coseni = Neumann

La FFT usa autovettori $v_j=e^{2\pi i kj/n}$ (esponenziali **periodiche**): impone la BC **periodica**.
La **DCT-II** usa autovettori $v_j=\cos\!\big(\tfrac{\pi k(j+1/2)}{n}\big)$ (coseni): questi hanno
**derivata nulla ai bordi** (funzione pari per riflessione a specchio), che è **esattamente** la BC di
**Neumann** $\partial\psi/\partial n=0$. Cambiando base cambiano gli autovalori del Laplaciano 1D:

$$\lambda^{\text{per}}_k=2\cos\tfrac{2\pi k}{n}-2\ \ (\text{FFT}),\qquad
\lambda^{\text{Neu}}_k=2\cos\tfrac{\pi k}{n}-2\ \ (\text{DCT}).$$

Il $\pi$ invece di $2\pi$ viene dal fatto che la **riflessione pari raddoppia** il dominio (periodo
$2n$), **dimezzando** la frequenza fondamentale. In 2D si somma sulle due direzioni. Tutto il resto
(RHS $\rho$, gauge DC, struttura $A^\top A$) è identico: **la scelta della trasformata è la scelta della
condizione al contorno** (esponenziali→periodico, coseni→Neumann, seni→Dirichlet).

### F.2 — Esempio numerico esplicito: $N=4$, BC di Neumann (DCT)

Con **Neumann** (flusso nullo ai bordi) la matrice **non** si richiude sul toro: ai due estremi manca
il vicino esterno, quindi le righe di bordo usano $-1$ invece di $-2$ (il pixel di bordo "vede" un solo
vicino):
$$L_{\text{Neu}}=\begin{pmatrix}-1&1&0&0\\ 1&-2&1&0\\ 0&1&-2&1\\ 0&0&1&-1\end{pmatrix}.$$
Gli autovettori sono i **coseni** DCT-II $v^{(k)}_j=\cos\!\big(\frac{\pi k(j+1/2)}{4}\big)$, e gli
autovalori $\lambda_k=2\cos\frac{\pi k}{4}-2$:

| $k$ | $2\cos(\pi k/4)-2$ | $\lambda_k$ |
|---|---|---|
| 0 | $2\cdot 1-2$ | $\mathbf{0}$ |
| 1 | $2\cdot 0.707-2$ | $-0.586$ |
| 2 | $2\cdot 0-2$ | $-2$ |
| 3 | $2\cdot(-0.707)-2$ | $-3.414$ |

**Verifica** su $v^{(2)}\propto(1,-1,-1,1)$ (dai coseni con $k=2$). Riga 0 di $L_{\text{Neu}}$
(stencil $[-1,1,0,0]$): $-1\cdot 1+1\cdot(-1)=-2=\lambda_2\cdot 1$. ✓ Riga 1 (stencil $[1,-2,1,0]$):
$1\cdot 1-2\cdot(-1)+1\cdot(-1)=2=\lambda_2\cdot(-1)$. ✓ Quindi $\lambda_2=-2$, come da formula
$2\cos(\pi\cdot 2/4)-2=2\cos(\pi/2)-2=-2$.

**Confronto FFT vs DCT sullo stesso $N=4$.** Stesso $\lambda_0=0$ (la costante è sempre nel nullo), ma
gli altri differiscono: FFT $\{0,-2,-4,-2\}$ (frequenze $2\pi k/N$), DCT $\{0,-0.586,-2,-3.414\}$
(frequenze $\pi k/N$, **dimezzate** dalla riflessione). La corona periodica $-2/-4/-2$ è simmetrica
perché $+k$ e $-k$ coincidono sul toro; la DCT no, perché il dominio riflesso non ha quella simmetria.

### F.3 — Dal 1D al 2D: gli autovalori si sommano

In 2D il Laplaciano è $\nabla^2=\partial_{xx}+\partial_{yy}$: la parte in $x$ e quella in $y$ agiscono su
indici diversi e **commutano**. Gli autovettori 2D sono i **prodotti** dei modi 1D
$v^{(k,l)}_{j,p}=v^{(k)}_j\,v^{(l)}_p$, e gli autovalori si **sommano** (somma di Kronecker):
$$\lambda_{k,l}=\lambda^{x}_k+\lambda^{y}_l.$$
È la riga di codice `_denom = 2cos(2πk/n) + 2cos(2πl/m) - 4`: i due $-2$ (uno per direzione) fanno il
$-4$. Esempio ($n=m=4$, FFT): $\lambda_{1,2}=(-2)+(-4)=-6$; $\lambda_{0,0}=0+0=0$ (il DC, di nuovo il
nullo). Idem per la DCT con $2\cos(\pi k/n)-2$. Ecco perché `_denom` è una **matrice** $n\times m$: un
autovalore per ogni coppia di frequenze $(k,l)$.

---

## Appendice G — Condizioni al contorno: Neumann, periodiche, padding, solvibilità

### Cosa sono le condizioni al contorno, e quale ci serve

Una **condizione al contorno** (BC) specifica come si comporta la soluzione **sul bordo** del dominio:
è necessaria perché il Laplaciano da solo non determina $\psi$ in modo unico (manca la costante, e serve
sapere cosa fa la soluzione ai lati). Le due che ci interessano:

- **Neumann** (flusso assegnato): $\partial\psi/\partial n=\mathbf g\cdot\hat{\mathbf n}$ — dominio
  **isolato**, niente flusso spurio dai bordi. È la BC **naturale** del nostro variazionale
  (Appendice A) e quella **fisicamente corretta** (l'immagine sta in un bordo scuro, non si ripete).
- **Periodica**: il bordo destro è "incollato" al sinistro (dominio a **toro**). È la BC implicita della
  **FFT**.

**Prima:** risolvevamo con la **FFT** ⇒ BC **periodica**, pur sapendo che quella giusta è Neumann —
scelta di **velocità** ($O(N\log N)$). **Ora:** la **DCT-II** diagonalizza il Laplaciano con **Neumann**
allo **stesso** costo $O(N\log N)$ (è una FFT su segnale simmetrizzato), quindi possiamo avere la BC
**corretta gratis** e **senza padding di guardia**. Perché "coseni = Neumann":
[Appendice F](#appendice-f--autovaloriautovettori-con-le-esponenziali-immaginarie). Confronto empirico
periodico↔Neumann: [dct_vs_fft.py](dct_vs_fft.py).

### Solvibilità (condizione di compatibilità di Neumann)

**Perché esiste una condizione.** Il problema di Neumann $\nabla^2\psi=f$ in $\Omega$,
$\partial\psi/\partial n=h$ sul bordo, ha l'operatore **singolare**: la funzione **costante** è nel suo
nucleo ($\nabla^2\,\text{cost}=0$, $\partial\,\text{cost}/\partial n=0$). Per l'**alternativa di
Fredholm**, un problema con operatore singolare è risolubile **solo se** il termine noto è ortogonale al
nucleo — qui il nucleo sono le costanti, quindi la condizione è un **unico vincolo scalare** sugli
integrali dei dati.

**La condizione (compatibilità).** Si ricava integrando la PDE su $\Omega$ e usando il teorema della
divergenza ($\nabla^2\psi=\nabla\cdot\nabla\psi$):
$$\int_\Omega f\,dA=\int_\Omega\nabla^2\psi\,dA=\oint_{\partial\Omega}\frac{\partial\psi}{\partial n}\,ds=\oint_{\partial\Omega}h\,ds
\quad\Longrightarrow\quad \boxed{\int_\Omega f\,dA=\oint_{\partial\Omega}h\,ds}.$$
La sorgente interna deve bilanciare il flusso imposto al bordo. (Riferimenti: Courant–Hilbert vol. 1,
cap. IV; Evans, *PDE*, §2.2, problema di Neumann.)

**Perché nel NOSTRO caso è automatica.** Da noi $f=\rho=\nabla\cdot\mathbf g$ e
$h=\mathbf g\cdot\hat{\mathbf n}$. La compatibilità diventa
$$\int_\Omega\nabla\cdot\mathbf g\,dA=\oint_{\partial\Omega}\mathbf g\cdot\hat{\mathbf n}\,ds,$$
che è **esattamente il teorema della divergenza applicato a $\mathbf g$**: un'**identità**, vera per
qualsiasi $\mathbf g$. Non è un vincolo da verificare — la nostra sorgente è la divergenza dello stesso
campo di cui $\mathbf g\cdot\hat{\mathbf n}$ è la componente normale, quindi la compatibilità è garantita
**per costruzione**.

**Unicità a meno di una costante.** Poiché la costante è nel nucleo, $\psi$ è determinata **a meno di
un offset additivo** (una fase globale, fisicamente irrilevante): la fissiamo con la gauge a media nulla.

**Caso periodico (FFT).** Sul toro non c'è bordo, $\oint=0$, quindi la condizione collassa a
$\int_\Omega\rho=0$. **Perché `rho_hat[0,0]=0`.** La componente DC è la somma di tutti i $\rho$; per il
teorema della divergenza discreto (telescopio della backward-difference) è il **flusso netto uscente**
di $\mathbf g$:

$$\hat\rho_{0,0}=\sum_{i,j}\rho_{i,j}=\sum_j(g_j-g_{j-1})=g_{\text{bordo dx}}-g_{\text{bordo sx}}=\text{flusso netto}.$$

Azzerarla ⇒ (a) impone la solvibilità $\int\rho=0$, (b) fissa la costante libera (lo spazio nullo del
Laplaciano: $\lambda_{0,0}=0$) scegliendo la soluzione a **media nulla**. `_denom[0,0]=1` serve solo a
evitare $0/0$: il valore è irrilevante perché il numeratore è già zero.

### Padding / bandlimit: due padding distinti (nota di consistenza con §5.4)

Attenzione a non confondere **due** cose che si chiamano "padding":

1. **Band-limit dell'SLM** (`F = np.pad(F, ((n//4,...)))`, $512\to768$): è l'oversampling di Chen che crea
   la griglia di lavoro (apertura SLM $384$, SR $512$, bordo = regione di rumore, maschere
   `bandlim_in`/`bandlim_ou`/`bandlim_spe`). Serve al loop GS, **non** alla BC di Poisson.
2. **Guard-band per Poisson**: un padding *aggiuntivo* attorno al ritaglio, che confinerebbe l'artefatto
   periodico (la "cucitura" del toro) in una fascia di guardia, per far sì che la FFT-periodica
   *approssimi* Neumann nella zona centrale.

**Cosa fa davvero il codice.** Il solve `irrotational_phase` gira sul **ritaglio $512$** e **non**
applica il guard-band (2): usa la FFT periodica e **tollera** la piccola cucitura ai bordi della SR (in
pratica trascurabile — è per questo che la FFT funzionava comunque bene). Il guard-band (2) resta quindi
un rimedio **teorico/opzionale**, non attivo in questo script.

**Con la DCT** il problema sparisce alla radice: la Neumann è imposta *esattamente* dallo specchio, senza
alcun padding di guardia (§5.5).

---

## Appendice H — Nota sperimentale: conteggio vortici vs momento angolare (OAM)

**Tentativo (fallito).** Abbiamo provato ad applicare lo stesso proiettore irrotazionale anche
sulla fase del **piano SLM** (il piano di Fourier dell'ologramma), non solo sul campo di arrivo,
nell'ipotesi che i vortici andassero "conservati/annichilati" anche lì durante la propagazione.
Script: [old/vah_projector_test_onslm.py](old/vah_projector_test_onslm.py).

**Esito: disastroso.** Sul piano SLM l'RMSE esplode e peggiora **monotonamente** con $\alpha$
(su `marmo.tif`: baseline $35.9$ → slm-proj $\alpha{=}0.5$ → $334$, $\alpha{=}1$ → $1271$), mentre
sul piano target funziona (RMSE $\sim 9$, vortici $\to 0$).

**Perché fallisce.** C'è una differenza fondamentale tra due quantità facili da confondere:

- **Momento angolare orbitale (OAM) totale** — quantità **integrale e pesata**
  $L_z\propto\int\operatorname{Im}(\psi^*\partial_\theta\psi)\,dA=\sum_\ell \ell\,|c_\ell|^2$.
  Sotto trasformata di Fourier / propagazione libera l'OAM **si conserva** (l'indice azimutale
  $\ell$ di un modo $e^{i\ell\theta}$ è preservato dalla FFT).
- **Numero di vortici** $N=N_++N_-$ — semplice **conteggio degli zeri** del campo. **Non si
  conserva**: i vortici nucleano e si annichilano in **coppie $\pm1$**, cambiando $N$ senza
  toccare né l'OAM né la carica netta $N_+-N_-$.

Quindi $N_{\text{SLM}}\neq N_{\text{target}}$ è **atteso** (nei dati: $\sim$36000 vs $\sim$7600):
i due piani sono legati da una FFT globale e i loro zeri non hanno corrispondenza. Sul piano SLM
quei $\sim$36000 vortici sono la **struttura speckle intrinseca dell'ologramma di Fourier** — il
*modo* in cui l'immagine è codificata nella fase, non difetti. Renderli irrotazionali **cancella
la codifica** e la ricostruzione collassa.

Inoltre nel loop GS **non** c'è propagazione *passiva*: i **vincoli di ampiezza** imposti in
entrambi i piani sono operazioni **non unitarie** che creano/distruggono vortici a ogni
iterazione, rompendo qualsiasi conservazione (anche dell'OAM). Una FFT "nuda" preserverebbe
l'OAM; le proiezioni di ampiezza no.

| Quantità | Conservata sotto FFT? | È il "numero di vortici"? |
|---|---|---|
| OAM totale $L_z$ | Sì (propagazione passiva) | No (integrale pesato) |
| Carica netta $N_+-N_-$ | In parte (contorno esterno) | No (differenza) |
| Conteggio $N_++N_-$ | **No** (coppie $\pm$ nascono/muoiono) | Sì |

**Conclusione operativa.** Il proiettore irrotazionale va applicato **solo sul campo di arrivo
(piano immagine)**, dove i vortici sono difetti reali della ricostruzione. Sul piano SLM no.

---

## Appendice I — Nota sperimentale: sweep del floor su target dark-heavy

**Problema.** Su target con molte zone buie (es. `Cat_black.tif`) il proiettore diventa
**estremamente rumoroso** e perde in RMSE contro il paper. Causa: dove l'intensità target è $\sim 0$
la fase è indefinita, quindi il gradiente wrapped lì è rumore; il Poisson solve è **globale** (una
sola FFT) e **spalma** quel rumore ovunque, contaminando anche le regioni luminose.

**Rimedio (leva principale): alzare il floor** `target_floor_rel`, che definisce la fase ovunque
e spegne il rumore alla sorgente. Script: [old/vah_ph_projector_floor_sweep_test.py](old/vah_ph_projector_floor_sweep_test.py).

**Risultati** (Cat_black, proiettore puro $\alpha=1$, 300 iter; `roughness` = jitter temporale
della curva RMSE, non contrasto spaziale):

| floor | proj RMSE | roughness | vortici | paper RMSE |
|---:|---:|---:|---:|---:|
| 0.001 | 20.7 | 0.95 | 9680 | 9.29 |
| 0.01 | 20.0 | 0.94 | 7486 | 6.66 |
| **0.03** | 5.70 | 0.055 | 18 | 5.75 |
| **0.1** | **4.96** | **0.013** | **0** | 6.56 |
| 0.3 | 6.16 | 0.014 | 0 | 10.18 |

**Lettura.** Da **floor $\approx 0.03$ in su** il proiettore **eguaglia/supera il paper** (di poco
a $0.03$, nettamente a $0.1$). Sotto $0.03$ è catastrofico (rumore da fase indefinita). A $0.3$
l'RMSE **risale**: un floor troppo alto è uno **sfondo diffuso esteso** che una fase irrotazionale
**non** sa ricostruire (stessa tensione dell'[Appendice H](#appendice-h--nota-sperimentale-conteggio-vortici-vs-momento-angolare-oam)).
**Finestra utile per questo target: floor $\sim 0.03$–$0.1$, sweet spot $\sim 0.1$.** Con $\alpha=1$
la transizione è un vero *cliff* tra $0.01$ e $0.03$; con $\alpha=0.5$ è più graduale.

---

## Appendice J — Nota sperimentale: strategia di alpha (costante vs schedulato)

**Domanda.** Fissato un buon floor, conviene un alpha **costante** o **schedulato** (forte
all'inizio poi 0? crescente? decrescente? impulsi al plateau)? Script:
[old/vah_projector_alpha_strategy_test.py](old/vah_projector_alpha_strategy_test.py).

**Esito.** Con un floor adeguato le strategie danno risultati **estremamente simili**: conviene
tenere un **alpha costante** (la scelta più semplice). Su `marmo.tif` (pieno, floor 5e-3, 400 iter)
tutte le strategie projector stanno a RMSE $\sim 7.5$–$8.5$ (vs off $35.7$, paper $12.5$); una
costante **$\sim 0.5$–$0.75$** è quasi-ottima **e** azzera i vortici.

Lo scheduling non aggiunge nulla di utile, anzi:

- **decrescente$\to 0$ / init-off:** RMSE ok, ma i **vortici ricompaiono** quando il projector si
  spegne tardi (es. `init_off_30` → 112 vortici, `cosine_down` → 46);
- **crescente:** strettamente **peggio** (perde la pulizia cruciale delle prime iterazioni), più rumoroso;
- **plateau_pulse (impulsi intermittenti):** il **più rumoroso** e il peggiore della famiglia projector.

Anche sul target difficile **Cat_black** (con floor $\sim 0.05$–$0.1$) le differenze tra strategie
sono **minime**.

**Conclusione operativa.** Fissa un buon floor (Appendice I) e usa un **alpha costante
$\sim 0.5$–$0.75$**, mantenuto fino alla fine. (Nota tecnica: il trigger del plateau va reso
**relativo**, $\text{mean}|\Delta\text{RMSE}|/\text{mean}(\text{RMSE})<\text{tol}$, altrimenti non
scatta mai quando l'RMSE è su scala assoluta grande.)

---

## Appendice K — Nota sperimentale: Poisson periodico/a-stagnazione e tempi

**Idea.** Applicare il proiettore **non** ogni iterazione ma solo **ogni tanto** — esattamente
dove il paper applica la sua annichilazione arctan2 — con trigger periodico (`ogni x`) o a
**stagnazione** (pendenza relativa dell'RMSE $<$ tol). Script:
[vah_projector_periodic_test.py](vah_projector_periodic_test.py).

**Risultato (marmo, x=50, 300 iter) — il vantaggio è il TEMPO.** A parità di schedule:

| metodo | RMSE | vortici | t/evento |
|---|---:|---:|---:|
| proj_periodic | 10.5 | 432 | **1.8 ms** |
| paper_periodic | 14.5 | 782 | **414.6 ms** |

Il proiettore è **migliore** (RMSE e vortici) e **$\sim$232× più veloce per annichilazione**: costo
$O(N\log N)$ contro $O(K\,N^2)$ del paper (una `arctan2` per vortice). Anche applicato **ogni**
iterazione (299 proiezioni, 2.68 s) è più veloce del paper applicato solo 5 volte (4.26 s), perché
ogni eliminazione del paper è costosissima. A **stagnazione** il divario cresce (più vortici al
trigger): $\sim$1.9 ms vs $\sim$786 ms/evento. **La velocità è il vantaggio principale del metodo.**

---

## Appendice L — Nota sperimentale: weighted/masked Poisson solve (accantonato)

**Idea.** Regolarizzazione spazialmente variabile guidata dall'intensità, in due forme **distinte**:
(a) **weighted solve** $w(x,y)\propto I$ che pesa il gradiente nel RHS del Poisson **prima** della
FFT (cambia $\psi$); (b) **masked application** $\alpha(x,y)=\alpha\cdot\text{mask}(x,y)$ che pesa
il blend (cambia *quanto* si mixa $\psi$, non $\psi$). Script:
[old/vah_projector_weighted_test.py](old/vah_projector_weighted_test.py).

**Esito: risultati instabili/insoddisfacenti → accantonato.** Su Cat_black (floor 0.01, $\alpha=0.5$):

| metodo | RMSE | roughness |
|---|---:|---:|
| plain | 10.57 | 0.244 |
| weighted_solve (a) | 11.05 | 0.012 |
| masked_alpha (b) | 14.83 | 0.401 |
| weighted+masked | 7.19 | 0.028 |
| paper | 6.66 | 0.252 |

(a) **stabilizza** (roughness $0.244\to0.012$) ma non migliora l'RMSE; (b) **da sola peggiora**;
solo (a)+(b) insieme si avvicinano al paper (7.19 vs 6.66, molto più liscio) ma il risultato è
**sensibile** alle soglie/softness → non robusto. Alzare il **floor** (Appendice I) è una leva più
semplice, forte e stabile. L'approccio con pesi resta come riferimento ma è **accantonato**.

---

## Appendice M — Inizializzazione: diffusore random “matched” alla signal region

**Contesto.** A convergenza l'errore residuo **non sono i vortici** (diagnosi: 0% dell'errore ai
core, ~82% nel segnale luminoso), ma il *floor* del phase-retrieval. L'unica leva che lo sfonda è
l'**inizializzazione di fase**.

**Cosa abbiamo trovato** (marmo):
- Init **random**: RMSE ~8.9 (semina ~87000 vortici, bacino di convergenza scadente).
- Init **quadratica** (lente, $\varphi=c(u^2+v^2)$): RMSE ~4.9, ma **target-dependent** (una lente
  tarata su target quadrato/uniforme) e non diversa.
- Init **random liscia** (diffusore): con i parametri giusti (sweep smoothness $\times$ ampiezza)
  **batte** la quadratica (fino a ~4.2), ma l'ottimo sembrava piccato e da tarare.

**Il criterio (principiato e target-independent).**
> Scegli la fase liscia tale che la **sua trasformata cada dentro la signal region**.

L'init è un **diffusore random**: il suo spread in campo lontano deve **riempire la SR** — non meno
(luce concentrata → speckle) né più (luce fuori dalla SR → persa). Lo spread ottimo è una quantità
**geometrica** ($N$ e $M$, non l'immagine) → **target-independent per costruzione**. Nel codice si
genera un campo liscio (low-pass di rumore bianco) e si **auto-tara l'ampiezza** (bisezione) finché
il raggio RMS del campo lontano di $e^{i\varphi}$ eguaglia $\sim (M/2)\cdot\text{fill}$; si verifica
che la **frazione di energia in-SR** sia alta. Empiricamente le celle migliori dello sweep
clusterizzano su $\text{amp}\cdot\text{ks}\approx\text{cost}$, coerente con "spread $\propto$ amp·ks =
dimensione SR".

**Risultato** (marmo, floor 5e-3; [vah_projector_smooth_init.py](vah_projector_smooth_init.py)):
diffusore matched RMSE **4.8–5.6** su 3 seed (best 4.81 $<$ quadratica 4.88), in-SR frac ~0.92,
ampiezza auto ~20, vortici finali ~0 — **senza brute force**. Essendo random è **diverso per seed**
→ **multiplexabile** (media incoerente di $N$ frame → speckle giù ancora). È il *best of both worlds*:
qualità della quadratica ma **principiato, target-independent e diverso**.

**Nota.** Su target dark-heavy (Cat_black) a floor basso il vantaggio dell'init si attenua, ma lì il
collo di bottiglia è il **floor** (Appendice I), non l'init.

---

## Appendice N — Dualità di Helmholtz: la parte rotazionale del paper

Questa appendice dimostra, passo-passo, che la fase-vortice del paper
$\varphi_{\text{vort}}=\sum_k q_k\,\theta_k$ è **esattamente** la parte rotazionale della decomposizione
di Helmholtz, e chiarisce la meccanica 2D ($\nabla^\perp$, rotore $=$ Laplaciano) usata nel testo
principale.

### N.1 — Stokes vs Helmholtz: due teoremi diversi

- **Stokes** non decompone: è la **contabilità della carica**. Lega la circuitazione al flusso del
  rotore racchiuso, $\oint_C\nabla\varphi\cdot d\boldsymbol\ell=\iint_S(\nabla\times\nabla\varphi)\,dA=2\pi\sum_{k\in S}q_k$.
  È lo strumento che *definisce* la carica; non fornisce la parte rotazionale.
- **Helmholtz** è la **decomposizione**: $\mathbf v=\nabla\psi+\nabla\times\mathbf A$, con la proprietà
  che i due pezzi sono separati dalle **sorgenti**: la divergenza sta tutta nell'irrotazionale, il
  rotore tutto nel solenoidale.

### N.2 — Perché in 2D "rotore di $\mathbf A$" = gradiente ruotato $\nabla^\perp$

Il campo è planare $\mathbf v=(v_x,v_y,0)$: il potenziale vettore punta fuori dal piano,
$\mathbf A=\chi(x,y)\,\hat{\mathbf z}$ (una sola *stream function* scalare, gli altri gradi di libertà
sono gauge). Allora

$$\nabla\times(\chi\hat{\mathbf z})=(\partial_y\chi,\ -\partial_x\chi)\equiv\text{gradiente di }\chi\text{ ruotato di }90^\circ.$$

Definiamo $\nabla^\perp\chi=(-\partial_y\chi,\partial_x\chi)$; è la stessa cosa a meno del segno di $\chi$
(orientazione). Non è un operatore nuovo: è la forma 2D di $\nabla\times\mathbf A$.

### N.3 — Perché il suo rotore è il Laplaciano (derivate pure vs miste)

Con $\mathbf v_{\text{rot}}=\nabla^\perp\chi=(-\partial_y\chi,\partial_x\chi)$:

$$\underbrace{\nabla\cdot(\nabla^\perp\chi)}_{\text{miste}}=-\partial_{xy}\chi+\partial_{yx}\chi=0,
\qquad
\underbrace{\nabla\times(\nabla^\perp\chi)}_{\text{pure}}=\partial_{xx}\chi+\partial_{yy}\chi=\nabla^2\chi.$$

Le derivate **miste** si cancellano (Schwartz), le **pure** si sommano → Laplaciano. La rotazione di
$90^\circ$ scambia i ruoli di divergenza e rotore:

| Campo | divergenza | rotore |
|---|---|---|
| $\nabla\psi$ (gradiente) | $\nabla^2\psi$ | $0$ |
| $\nabla^\perp\chi$ (gradiente ruotato) | $0$ | $\nabla^2\chi$ |

È **questo** che rende risolvibile la parte rotazionale con la **stessa** Poisson di quella
irrotazionale, ma con sorgente il **rotore** invece della divergenza.

### N.4 — Il paper: risolvere $\nabla^2\chi=$ vorticità con la funzione di Green

La vorticità di un gas di vortici puntiformi è
$$\rho=\nabla\times\nabla\varphi=2\pi\sum_k q_k\,\delta(\mathbf r-\mathbf r_k).$$
"Invertire il Laplaciano" = convolvere con la sua **funzione di Green** $G$, definita da
$\nabla^2 G=\delta$. In 2D, per simmetria radiale $\nabla^2G=\tfrac1r\tfrac{d}{dr}(r\,G')=0$ fuori
dall'origine dà $G=C\ln r+D$; la costante si fissa integrando su un dischetto e usando la divergenza
$\oint\nabla G\cdot\hat{\mathbf n}\,d\ell=2\pi C=1$, quindi
$$G(\mathbf r)=\frac{1}{2\pi}\ln|\mathbf r|.$$
Per sovrapposizione (convoluzione + proprietà setaccio della delta):
$$\chi=G*\rho=\int G(\mathbf r-\mathbf r')\,2\pi\sum_k q_k\delta(\mathbf r'-\mathbf r_k)\,d^2r'
=2\pi\sum_k q_k\,G(\mathbf r-\mathbf r_k)=\sum_k q_k\ln|\mathbf r-\mathbf r_k|,$$
(il $2\pi$ della sorgente cancella l'$\tfrac1{2\pi}$ della Green). Questa $\chi$ è la **stream function**.

### N.5 — Versione semplificata: $\theta=\operatorname{atan2}$ è un vortice, e la produttoria

In parallelo, la via "diretta" del paper: la fase di un singolo vortice unitario è
$\theta=\operatorname{atan2}(y,x)=\arg(x+iy)=\operatorname{Im}\log(x+iy)$. Sommando su tutti i core,
pesati per la carica:
$$\varphi_{\text{vort}}=\sum_k q_k\,\theta_k=\sum_k q_k\arg(z-z_k)=\operatorname{Im}\sum_k q_k\log(z-z_k)
=\operatorname{Im}\,\log\!\prod_k (z-z_k)^{q_k},\qquad z=x+iy.$$
Cioè: la fase vorticosa totale è la **parte immaginaria del logaritmo della produttoria** dei
$(z-z_k)^{q_k}$. È una fase **continua** (fuori dai core) e con l'avvolgimento **giusto** ($+2\pi q_k$)
attorno a ciascun core.

### N.6 — Il punto: questa fase È la parte rotazionale (non solo "assomiglia")

Le due vie coincidono. Usando $\partial_x\ln r=x/r^2$, $\partial_y\ln r=y/r^2$:
$$\nabla^\perp\ln|\mathbf r-\mathbf r_k|=\Big(-\tfrac{y-y_k}{r^2},\ \tfrac{x-x_k}{r^2}\Big)=\nabla\,\theta_k.$$
Quindi
$$\nabla^\perp\chi=\sum_k q_k\,\nabla^\perp\ln|\mathbf r-\mathbf r_k|=\sum_k q_k\,\nabla\theta_k
=\nabla\Big(\sum_k q_k\theta_k\Big)=\nabla\varphi_{\text{vort}}.$$
**Il gradiente della fase-vortice del paper è, punto per punto, la parte solenoidale
$\nabla^\perp\chi$ di Helmholtz.** Non è una fase "che assomiglia ai vortici": è **la soluzione
analitica esatta** di $\nabla^2\chi=$ vorticità. Verifica di consistenza: $\nabla\cdot\nabla\theta=0$
ovunque e $\nabla\times\nabla\theta=2\pi\delta$ (divergenza nulla + tutto il rotore), quindi
$\nabla\theta$ è **solenoidale** malgrado sia scritto come gradiente — perché $\theta$ è **multivalore**
(vortice puntiforme classico: potenziale $\theta$, stream function $\ln r$).

### N.7 — La dualità dei due metodi in una tabella

Stessa $\nabla^2$, stessa funzione di Green $\tfrac1{2\pi}\ln r$, sorgenti complementari:

| | Sorgente | Solve | Estrae | Azione |
|---|---|---|---|---|
| **Paper** | rotore $\nabla\times\mathbf v$ (le cariche) | $\chi=G*(\text{rotore})$, poi $\nabla^\perp\chi$ | parte **rotazionale** $\nabla\varphi_{\text{vort}}$ | **sottrae** $\varphi_{\text{vort}}$ |
| **Nostro** | divergenza $\nabla\cdot\mathbf v$ | $\psi=G*(\text{divergenza})$ (via FFT/DCT) | parte **irrotazionale** $\nabla\psi$ | **tiene** $\psi$ |

Nel continuo, con BC compatibili, $\varphi_{\text{free}}=\varphi-\varphi_{\text{vort}}=\psi$: **stesso
oggetto, due strade**. Differiscono in pratica per: detection (il paper deve localizzare i core; noi
no), BC/parte armonica (il $\ln r$ assume spazio libero; noi periodico o Neumann), costo ($O(KN^2)$ vs
$O(N\log N)$).

### N.8 — Perché il blend annichila (nullo di ampiezza)

Il winding intero non può passare da $1$ a $0$ con continuità **senza** che $|E|$ si annulli da qualche
parte. Il blend di campo $E=(1-t)e^{i\varphi}+t e^{i\psi}$ fornisce proprio quel nullo: scritto come
$E=e^{i\varphi}[(1-t)+t e^{i\delta}]$ con $\delta=\psi-\varphi$, attorno a un core $\delta$ avvolge di
$-2\pi$ (perché $\psi$ è privo di vortici), e il fattore $g=(1-t)+t e^{i\delta}$ accerchia l'origine
(winding $-1$, annichilazione) non appena $t>\tfrac12$, passando per $|g|=0$ a $t=\tfrac12$. È il motivo
per cui il blend va fatto **nel dominio del campo** e non delle fasi (che non ha nulli di ampiezza e può
solo *spostare* i core, non annichilarli).

---

## Riferimenti

- Gelfand & Fomin, *Calculus of Variations* (Dover) — Euler–Lagrange, natural BC.
- Courant & Hilbert, *Methods of Mathematical Physics*, vol. 1, cap. IV.
- Evans, *Partial Differential Equations* (AMS) — problema di Neumann e condizione di compatibilità (§2.2).
- Ghiglia & Pritt (1998), *Two-Dimensional Phase Unwrapping: Theory, Algorithms, and Software*, cap. 5.
- Strang, *Computational Science and Engineering* — $A^\top A$ e metodi spettrali.
- Codice: [`irrotational_phase`](vah_projector_fft_test.py) in questa cartella; funzioni condivise in
  [../1_Alternative_projection](../1_Alternative_projection).
