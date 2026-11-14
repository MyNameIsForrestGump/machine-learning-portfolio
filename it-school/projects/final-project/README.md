# LoRA Model Merging – Reducerea interferenței între modele prin adaptare de rang mic

## 1. Descrierea proiectului

Acest proiect investighează o problemă din domeniul **Deep Learning**, și anume combinarea mai multor modele specializate într-un singur model, fără pierderi semnificative de performanță.

În mod obișnuit, un model preantrenat poate fi adaptat pentru diferite sarcini folosind tehnica **LoRA (Low-Rank Adaptation)**, care permite antrenarea unui număr redus de parametri.

Problema apare atunci când dorim să combinăm două astfel de adaptări. Cea mai simplă metodă constă în adunarea actualizărilor parametrilor, însă această operație poate produce interferențe între sarcini și poate reduce performanța modelului final.

**Obiectivul proiectului** este să investigheze această problemă și să compare metoda clasică de combinare cu o abordare bazată pe regularizare, care încearcă să reducă influența unui adaptor asupra datelor asociate celuilalt.

Proiectul pornește de la o investigație matematică a nucleului matricilor de rang mic și urmărește verificarea experimentală a ideii.

## 2. Context și aplicabilitate practică

Să presupunem că o companie utilizează un model de inteligență artificială pentru mai multe sarcini.

În loc să antreneze și să păstreze câte un model complet pentru fiecare sarcină, compania poate utiliza un model preantrenat comun și adaptoare LoRA specializate.

De exemplu, un model lingvistic ar putea fi adaptat pentru:

- Clasificarea sentimentului exprimat într-un text.
- Identificarea propozițiilor cu semnificații similare.

Ideal, am dori să combinăm cele două adaptări într-un singur model, păstrând performanța fiecăreia.

Totuși, adunarea actualizărilor nu garantează păstrarea predicțiilor inițiale.

Acest proiect analizează dacă o penalizare introdusă în timpul antrenării poate reduce efectele nedorite ale combinării.

## 3. Fundamente matematice

### 3.1. Low-Rank Adaptation (LoRA)

Considerăm o matrice de ponderi a unui model preantrenat:

\[
W_0\in\mathbb R^{m\times n}
\]

În loc să modificăm toți parametrii matricei, LoRA introduce o actualizare de rang mic:

\[
W=W_0+\Delta W
\]

unde:

\[
\Delta W=BA
\]

cu:

\[
A\in\mathbb R^{r\times n},\qquad
B\in\mathbb R^{m\times r}
\]

și \(r\ll\min(m,n)\).

Această factorizare reduce numărul parametrilor antrenabili.

### 3.2. Problema combinării adaptoarelor

Presupunem că avem două sarcini, pentru care antrenăm separat:

\[
\Delta W_1=B_1A_1
\]

\[
\Delta W_2=B_2A_2
\]

Metoda clasică de combinare utilizează:

\[
W_{\text{merged}}=W_0+\Delta W_1+\Delta W_2
\]

Pentru o reprezentare \(h_1\) aparținând primei sarcini, contribuția suplimentară introdusă de al doilea adaptor este:

\[
\Delta W_2h_1
\]

Analog, pentru o reprezentare \(h_2\) din a doua sarcină, contribuția primului adaptor este:

\[
\Delta W_1h_2
\]

Ne dorim ca aceste contribuții să fie cât mai mici, fără să afectăm performanța adaptoarelor pe sarcinile pentru care au fost antrenate.

### 3.3. Legătura cu nucleul unei matrice

Deoarece:

\[
\operatorname{rank}(\Delta W)\le r
\]

din teorema rang–nulitate rezultă:

\[
\dim\ker(\Delta W)\ge n-r
\]

Prin urmare, o actualizare LoRA cu rang mic are un nucleu de dimensiune mare.

Această observație motivează ideea de a încuraja reprezentările unei sarcini să fie transformate în vectori de normă mică de către adaptorul celeilalte sarcini.

Nu impunem însă apartenența exactă la nucleu, deoarece o asemenea constrângere ar putea forța actualizarea să devină matricea nulă atunci când reprezentările generează întreg spațiul.

## 4. Metoda propusă – BA Regularization

Pentru a reduce interferența dintre adaptoare, introducem o penalizare suplimentară în funcția de pierdere.

Funcția obiectiv este:

\[
L_{BA}=\frac12(L_1+L_2)+\lambda R_{BA}
\]

unde \(L_1\) și \(L_2\) reprezintă pierderile corespunzătoare celor două sarcini.

Termenul de regularizare este:

\[
R_{BA}=\frac12\left[
\mathbb E_{h_1}\operatorname{mean}((B_2A_2h_1)^2)
+
\mathbb E_{h_2}\operatorname{mean}((B_1A_1h_2)^2)
\right]
\]

Acesta penalizează efectul unui adaptor asupra reprezentărilor celuilalt task.

Parametrul \(\lambda\) controlează intensitatea regularizării.

**Intuiția metodei:** fiecare adaptor trebuie să învețe să rezolve propria sarcină, producând în același timp modificări cât mai mici asupra reprezentărilor celeilalte sarcini.

### Metode comparate

În cadrul experimentelor au fost analizate trei strategii:

1. **Naive Merging** – antrenarea adaptoarelor fără regularizare suplimentară, urmată de adunarea actualizărilor.
2. **A-only Regularization** – penalizarea efectului factorului \(A\) asupra reprezentărilor celuilalt task.
3. **BA Regularization** – penalizarea efectului complet al actualizării \(BA\), metoda investigată în proiect.

## 5. Dataset și preprocesare

### Experimentul principal – Handwritten Digits

A fost utilizat datasetul public **Digits**, disponibil în biblioteca scikit-learn.

Datasetul conține 1.797 de imagini reprezentând cifre scrise de mână, din cele 10 clase (0–9).

Fiecare imagine are dimensiunea \(8\times8\) pixeli, fiind reprezentată prin 64 de caracteristici numerice.

Sursă: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

Pentru a construi două sarcini distincte, imaginile originale au fost transformate astfel:

- **Task 1:** deplasarea ciclică a imaginilor cu doi pixeli pe verticală.
- **Task 2:** deplasarea ciclică a imaginilor cu doi pixeli pe orizontală.

Etichetele originale ale cifrelor au fost păstrate.

Datele au fost împărțite în:

- 60% antrenare;
- 20% validare;
- 20% testare.

Pentru analiza exploratorie sunt relevante distribuția claselor, exemplele de imagini originale și transformate, distribuția valorilor pixelilor și comparația vizuală dintre cele două sarcini.

## 6. Arhitectura modelului

Pentru experimentul principal a fost utilizat un model **Multilayer Perceptron (MLP)**.

Arhitectura include:

- 64 de caracteristici de intrare;
- un strat ascuns cu 128 de neuroni;
- un strat de ieșire pentru clasificarea celor 10 cifre.

Modelul de bază a fost antrenat pe imaginile originale.

Ulterior, parametrii modelului de bază au fost înghețați, iar pentru fiecare sarcină au fost antrenate adaptoare LoRA de rang 8, aplicate primului strat liniar.

Au fost comparate performanțele adaptoarelor individuale și ale modelului obținut după combinarea lor.

## 7. Antrenare și evaluare

Experimentul principal a fost repetat folosind cinci seed-uri diferite pentru a analiza variabilitatea rezultatelor.

Valoarea parametrului de regularizare \(\lambda\) a fost selectată folosind performanța pe setul de validare.

Evaluarea din raportul experimental utilizează două măsurători principale:

**Individual Accuracy** – media acurateților obținute cu adaptoarele evaluate separat.

**Merged Accuracy** – media acurateților obținute după combinarea adaptoarelor.

Diferența:

\[
D_{\text{merge}}=
A_{\text{individual}}-A_{\text{merged}}
\]

măsoară pierderea medie de performanță produsă de combinare.

Pentru versiunea completă a proiectului practic vor fi incluse și Precision, Recall, F1-Score, confusion matrix și graficele evoluției antrenării.

## 8. Rezultate experimentale

### 8.1. Experimentul pe cifre

Rezultatele reprezintă media și deviația standard obținute pe cinci seed-uri.

| Metodă | Individual Accuracy | Merged Accuracy |
|---|---:|---:|
| Naive | 92,75% ± 1,59 | 51,03% ± 6,99 |
| A-only | 91,92% ± 2,76 | 88,97% ± 1,56 |
| BA | 92,39% ± 0,79 | 90,75% ± 0,66 |

Metoda clasică de combinare produce o scădere importantă a performanței, de la 92,75% la 51,03%.

În schimb, metoda BA păstrează o acuratețe medie de 90,75% după combinare.

Aceasta reprezintă o îmbunătățire de **39,72 puncte procentuale** față de Naive Merging.

Rezultatele sugerează că regularizarea propusă poate limita degradarea produsă de combinarea adaptoarelor în acest experiment controlat.

### 8.2. Experiment suplimentar – BERT-tiny

Pentru a investiga aplicabilitatea metodei asupra unui model Transformer preantrenat, au fost efectuate experimente suplimentare cu BERT-tiny pe două dataseturi GLUE:

- SST-2 – clasificarea sentimentului.
- MRPC – clasificarea perechilor de propoziții în funcție de relația de parafrazare.

Au fost utilizate adaptoare LoRA de rang 4 în proiecțiile query și value ale mecanismului de atenție.

Rezultatele după combinarea adaptoarelor au fost:

| Metodă | Merged Accuracy |
|---|---:|
| Naive | 71,15% |
| A-only | 70,97% |
| BA | 71,19% |

În acest experiment, metoda BA nu a produs o îmbunătățire semnificativă față de metoda clasică.

Un motiv important este că metoda Naive Merging nu a produs o degradare sistematică a performanței în această configurație.

De asemenea, penalizarea BA a redus magnitudinea actualizărilor dintre taskuri cu aproximativ 48,6%, dar a redus și magnitudinea unor actualizări utile.

Prin urmare, reducerea interferenței măsurate nu garantează automat îmbunătățirea performanței de clasificare.

## 9. Interpretarea rezultatelor

Experimentele evidențiază faptul că eficiența regularizării depinde de caracteristicile modelului și ale sarcinilor.

Pe datasetul Digits, unde combinarea naivă produce o degradare considerabilă, metoda BA oferă o îmbunătățire importantă.

Pe BERT-tiny, unde actualizările celor două adaptoare nu produc o degradare semnificativă după combinare, beneficiile regularizării nu sunt evidente.

O limitare a metodei este fenomenul de **shrinkage**: penalizarea poate reduce nu doar interferența nedorită, ci și amplitudinea actualizărilor utile.

Aceste rezultate arată că este important să evaluăm atât performanța finală, cât și mecanismele interne ale modelului.

## 10. Limitări și dezvoltări viitoare

Experimentele sunt preliminare și nu demonstrează superioritatea generală a metodei BA față de alte metode de model merging.

Printre direcțiile de dezvoltare se numără:

- Evaluarea pe mai multe modele și perechi de taskuri.
- Compararea cu metode de merging publicate.
- Analiza geometrică a subspațiilor reprezentărilor prin SVD.
- Investigarea unghiurilor principale dintre subspațiile celor două taskuri.
- Dezvoltarea unei penalizări relative, care să reducă interferența fără să micșoreze excesiv actualizările utile.
- Evaluarea modificărilor predicțiilor finale, nu doar a actualizărilor interne.

Aceste direcții sunt propuneri pentru experimente viitoare, nu rezultate deja validate.

## 11. Tehnologii utilizate

- Python
- PyTorch
- NumPy
- scikit-learn
- Hugging Face Transformers
- Google Colab

## 12. Concluzii

Acest proiect investighează o problemă relevantă pentru adaptarea eficientă a modelelor de Deep Learning: combinarea mai multor actualizări LoRA într-un singur model.

Pornind de la proprietățile matricilor de rang mic și de la teorema rang–nulitate, a fost investigată o funcție de regularizare care penalizează influența unui adaptor asupra reprezentărilor celuilalt task.

Experimentele pe cifre scrise de mână au arătat că această abordare poate reduce semnificativ pierderea de performanță după combinare.

În același timp, experimentele pe BERT-tiny au evidențiat limitele metodei și necesitatea unei evaluări pe sarcini și arhitecturi diferite.

**Contribuția principală a proiectului constă în investigarea experimentală a unei strategii de regularizare inspirate din algebra liniară, care urmărește reducerea interferenței dintre adaptoarele LoRA.**

---

**Autor:** Mihnea-Ionuț Marinescu

**Proiect final:** IT School – Machine Learning

**Lucrare asociată:** *Kernel-Aware Low-Rank Adaptation for Interference-Minimized LoRA Merging: Preliminary Experimental Results* (2026)
