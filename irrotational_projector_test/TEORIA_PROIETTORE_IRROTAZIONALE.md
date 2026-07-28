# Proiettore Irrotazionale — Teoria e Metodo

Questo documento raccoglie **tutti** i passaggi teorici e numerici del metodo del "proiettore
irrotazionale" (least-squares + Poisson via FFT) che sperimentiamo in questa cartella, come
alternativa veloce all'annichilazione dei vortici del paper.

Struttura: una parte principale **breve** (§1–§5) che segue il filo logico dal problema fisico
fino alle tre righe di codice, e una serie di **appendici** (A–G) con tutte le derivazioni di
dettaglio (calcolo delle variazioni, aggiunto/trasposto, Laplaciano convolutivo, autovalori con
esponenziali immaginarie, condizioni al contorno).

Codice di riferimento: la funzione [`irrotational_phase`](vah_ph_projector_test.py) negli script di
questa cartella. Le funzioni condivise di rilevamento/eliminazione vortici stanno nella cartella
sorella: [function_vortex_detection_accegpu.py](../1_Alternative_projection/function_vortex_detection_accegpu.py)
e [function_vortex_elimination_accegpu.py](../1_Alternative_projection/function_vortex_elimination_accegpu.py).

---

## Indice

- [1. Cos'è un vortice ottico (formalismo del paper)](#1-cosè-un-vortice-ottico-formalismo-del-paper)
- [2. Come si rilevano i vortici: paper vs nostro metodo](#2-come-si-rilevano-i-vortici-paper-vs-nostro-metodo)
- [3. Formalizzazione: il problema di minimo (Lagrange/least-squares)](#3-formalizzazione-il-problema-di-minimo-lagrangeleast-squares)
- [4. Dimostrazione: dal minimo all'equazione di Poisson](#4-dimostrazione-dal-minimo-allequazione-di-poisson)
- [5. Come si risolve Poisson: analitico e numerico (FFT)](#5-come-si-risolve-poisson-analitico-e-numerico-fft)
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

---

## 1. Cos'è un vortice ottico (formalismo del paper)

Un campo ottico è $h(\mathbf r)=a(\mathbf r)\,e^{i\varphi(\mathbf r)}$. Un **vortice ottico**
(singolarità di fase) è un punto dove l'ampiezza si annulla, $a=0$, e la fase è indefinita:
girando attorno al punto, la fase avvolge di $\pm 2\pi$. La **carica topologica** è

$$Q=\frac{1}{2\pi}\oint_C \nabla\varphi\cdot d\boldsymbol\ell \in\{\dots,-1,0,+1,\dots\}.$$

**Punto cruciale:** in presenza di vortici $\nabla\varphi$ **non** è il gradiente di una funzione a
valore singolo. Per la **decomposizione di Helmholtz** ogni campo si scrive come

$$\nabla\varphi=\underbrace{\nabla\psi}_{\text{irrotazionale}}+\underbrace{\nabla\times\mathbf A}_{\text{rotazionale (vortici)}},$$

dove la parte irrotazionale $\nabla\psi$ ha $\nabla\times\nabla\psi=0$ (nessun vortice), mentre la
parte rotazionale porta tutta la carica: $\nabla\times(\nabla\varphi)=2\pi\sum_k q_k\,\delta(\mathbf r-\mathbf r_k)$.

**Obiettivo di entrambi i metodi:** estrarre $\psi$, il "potenziale scalare" privo di vortici.

---

## 2. Come si rilevano i vortici: paper vs nostro metodo

**Rilevamento (comune a entrambi).** Si discretizza $Q$ su un anello di 4 pixel (plaquette). Con
il gradiente *wrapped* $W(t)=\mathrm{mod}(t+\pi,2\pi)-\pi$:

$$g_x=W(\Delta_x\varphi),\quad g_y=W(\Delta_y\varphi),\qquad
\text{curl}[i,j]=g_x[i,j]+g_y[i,j{+}1]-g_x[i{+}1,j]-g_y[i,j].$$

Se $\text{curl}\approx+2\pi$ → vortice positivo; se $\approx-2\pi$ → negativo (soglia `2π − 0.1`).
Questo è il cuore di [function_vortex_detection_accegpu.py](../1_Alternative_projection/function_vortex_detection_accegpu.py).

**Metodo del paper (arctan2 / funzione di Green).** Note posizioni $\mathbf r_k$ e cariche $q_k$, si
ricostruisce esplicitamente la parte rotazionale come somma delle "viti" canoniche e la si sottrae:

$$\varphi_{\text{vort}}(\mathbf r)=\sum_k q_k\arg(z-z_k)=\operatorname{Im}\,\log\!\prod_k(z-z_k)^{q_k},
\qquad \varphi_{\text{free}}=(\varphi-\varphi_{\text{vort}})\bmod 2\pi.$$

Costo $O(K\,N^2)$ (un `arctan2` per vortice su tutta la griglia): esatto e fisicamente
trasparente, ma dipende dalla soglia di rilevamento e diventa lento quando i vortici $K$ sono molti.

**Il nostro metodo (least-squares → Poisson via FFT).** Non localizziamo i vortici. Calcoliamo
direttamente la parte irrotazionale come **proiezione ai minimi quadrati** del gradiente misurato,
risolvendo un'equazione di Poisson con **una singola FFT**: costo $O(N\log N)$, **indipendente dal
numero di vortici**. È la funzione [`irrotational_phase`](vah_ph_projector_test.py).

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

## 5. Come si risolve Poisson: analitico e numerico (FFT)

### Analitico

Il Laplaciano è un operatore **invariante per traslazione** ⇒ diagonale nella base di Fourier. Con le
esponenziali $e^{i\mathbf k\cdot\mathbf r}$ come autofunzioni:

$$\nabla^2 e^{i\mathbf k\cdot\mathbf r}=-|\mathbf k|^2\,e^{i\mathbf k\cdot\mathbf r}
\quad\Longrightarrow\quad \hat\psi(\mathbf k)=\frac{\hat\rho(\mathbf k)}{-|\mathbf k|^2}.$$

Cioè: trasformo, divido per $-|\mathbf k|^2$, antitrasformo. (Dettaglio autovalori: [Appendice F](#appendice-f--autovaloriautovettori-con-le-esponenziali-immaginarie);
carattere convolutivo: [Appendice E](#appendice-e--il-laplaciano-come-operatore-convolutivo).)

### Numerico (una singola FFT)

Sulla griglia periodica $n\times m$ lo stencil a 5 punti ha autovalori = **FFT del kernel**:

$$\lambda_{k,l}=2\cos\frac{2\pi k}{n}+2\cos\frac{2\pi l}{m}-4\;\approx\;-|\mathbf k|^2\ \text{(basse frequenze)}.$$

I tre passi $\psi=\mathcal F^{-1}\big[\hat\rho/\lambda\big]$ sono le tre righe centrali di
[`irrotational_phase`](vah_ph_projector_test.py):

```python
# eigenvalues of the 5-point Laplacian (periodic BC), precomputed once
_denom = 2*cp.cos(2*cp.pi*_ii/n) + 2*cp.cos(2*cp.pi*_jj/m) - 4
_denom[0, 0] = 1.0                     # avoid 0/0 on the DC mode

rho_hat = cp.fft.fft2(rho)             # F{rho}
rho_hat[0, 0] = 0.0                    # solvability + zero-mean (see Appendix G)
psi = cp.real(cp.fft.ifft2(rho_hat / _denom))   # F^{-1}{ rho_hat / lambda }
```

- **`_denom`** è la diagonale degli autovalori (FFT del kernel), non diagonale nella base dei pixel ma
  diagonale nella base di Fourier.
- **`rho_hat[0,0]=0`** impone media nulla / solvibilità (il modo DC ha $\lambda_{0,0}=0$): vedi
  [Appendice G](#appendice-g--condizioni-al-contorno-neumann-periodiche-padding-solvibilità).
- **`cp.real`** scarta la parte immaginaria residua ($\sim10^{-16}$): $\psi$ è reale per simmetria
  hermitiana (vedi nota in Appendice F).

Costruzione del termine noto `rho` (divergenza discreta del gradiente wrapped) e stencil: [Appendice B](#appendice-b--dal-continuo-al-numerico-vettorizzazione-stencil-ata).

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

Corrispondenza col codice di [`irrotational_phase`](vah_ph_projector_test.py):

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

**Nota (perché $\psi$ è reale).** $\rho$ reale ⇒ $\hat\rho$ hermitiano; $\lambda_{k,l}$ è reale e pari
($\lambda_{-k,-l}=\lambda_{k,l}$), quindi $\hat\rho/\lambda$ resta hermitiano e la sua IFFT è reale.
`cp.real(...)` scarta solo il residuo numerico ($\sim10^{-16}$) e converte il tipo complesso→float.

---

## Appendice G — Condizioni al contorno: Neumann, periodiche, padding, solvibilità

### Il problema "vero" è Neumann; risolviamo come periodico

Il variazionale produce spontaneamente la BC **naturale** di Neumann
$\partial\psi/\partial n=\mathbf g\cdot\hat{\mathbf n}$. Ma per usare la FFT (che diagonalizza il
Laplaciano solo con periodicità) risolviamo come se il dominio fosse **periodico** (toro). È una scelta
di **velocità** ($O(N\log N)$ vs $O(N^2)$/$O(N^3)$), non di fisica.

### Solvibilità = flusso netto nullo

La condizione generale è $\int_\Omega\rho=\oint_{\partial\Omega}\partial\psi/\partial n$. Con dato
Neumann $=\mathbf g\cdot\hat{\mathbf n}$ diventa $\int\rho=\oint\mathbf g\cdot\hat{\mathbf n}$, che è
**automatica** perché $\rho=\nabla\cdot\mathbf g$ (teorema della divergenza). Sul toro non c'è bordo
($\oint=0$), quindi collassa a $\int\rho=0$.

**Perché `rho_hat[0,0]=0`.** La componente DC è la somma di tutti i $\rho$; per il teorema della
divergenza discreto (telescopio della backward-difference) è il **flusso netto uscente** di $\mathbf g$:

$$\hat\rho_{0,0}=\sum_{i,j}\rho_{i,j}=\sum_j(g_j-g_{j-1})=g_{\text{bordo dx}}-g_{\text{bordo sx}}=\text{flusso netto}.$$

Azzerarla ⇒ (a) impone la solvibilità $\int\rho=0$, (b) fissa la costante libera (lo spazio nullo del
Laplaciano: $\lambda_{0,0}=0$) scegliendo la soluzione a **media nulla**. `_denom[0,0]=1` serve solo a
evitare $0/0$: il valore è irrilevante perché il numeratore è già zero.

### Padding / bandlimit (perché $768$ vs $512$)

Cambiare Neumann→periodico modifica leggermente il problema vicino ai bordi (il toro "ricollega" i lati
opposti). Il **padding** ($n/4$ per lato: $512\to768$) confina l'artefatto periodico nella fascia di
guardia, così la regione di interesse centrale $512\times512$ — quella che si ritaglia e si usa —
resta sostanzialmente identica alla soluzione Neumann. Le maschere `bandlim_in`/`bandlim_ou` separano
regione utile e fascia, `bandlim_spe` limita la banda nello spazio spettrale.

---

## Appendice H — Nota sperimentale: conteggio vortici vs momento angolare (OAM)

**Tentativo (fallito).** Abbiamo provato ad applicare lo stesso proiettore irrotazionale anche
sulla fase del **piano SLM** (il piano di Fourier dell'ologramma), non solo sul campo di arrivo,
nell'ipotesi che i vortici andassero "conservati/annichilati" anche lì durante la propagazione.
Script: [vah_projector_test_onslm.py](vah_projector_test_onslm.py).

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
e spegne il rumore alla sorgente. Script: [vah_ph_projector_floor_sweep_test.py](vah_ph_projector_floor_sweep_test.py).

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
[vah_projector_alpha_strategy_test.py](vah_projector_alpha_strategy_test.py).

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
[vah_projector_weighted_test.py](vah_projector_weighted_test.py).

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

## Riferimenti

- Gelfand & Fomin, *Calculus of Variations* (Dover) — Euler–Lagrange, natural BC.
- Courant & Hilbert, *Methods of Mathematical Physics*, vol. 1, cap. IV.
- Ghiglia & Pritt (1998), *Two-Dimensional Phase Unwrapping: Theory, Algorithms, and Software*, cap. 5.
- Strang, *Computational Science and Engineering* — $A^\top A$ e metodi spettrali.
- Codice: [`irrotational_phase`](vah_ph_projector_test.py) in questa cartella; funzioni condivise in
  [../1_Alternative_projection](../1_Alternative_projection).
