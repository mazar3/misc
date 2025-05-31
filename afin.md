**Exercice 1**

Soit un emprunt remboursable par annuités constantes. Les données pour la 10e année sont :
*   Amortissement : $$R_{10} = 3\,984,31€$$
*   Intérêt : $$I_{10} = 8\,573,26€$$
*   Le "Solde restant dû" fourni, 73 954,41€, est interprété comme $$S_{10}$$ (solde après paiement de la 10e annuité), conformément à la structure typique d'un tableau d'amortissement (cf. slide 87 du cours).

Le solde restant dû avant le paiement de la 10e annuité ($$S_9$$) est donc :
$$S_9 = S_{10} + R_{10} = 73\,954,41€ + 3\,984,31€ = 77\,938,72€$$.

L'annuité constante $$A$$ est :
$$A = I_{10} + R_{10} = 8\,573,26€ + 3\,984,31€ = 12\,557,57€$$.

**a) Que vaut le taux d'intérêt ? Quelle est la durée de l'emprunt ?**

*   **Taux d'intérêt ($$i$$)**
    L'intérêt de la 10e période $$I_{10}$$ est calculé sur le solde restant dû au début de cette période, $$S_9$$.
    $$I_{10} = S_9 \times i$$
    $$i = \frac{I_{10}}{S_9} = \frac{8\,573,26€}{77\,938,72€} = 0,11$$
    Le taux d'intérêt annuel est de $$11\%$$.

*   **Durée de l'emprunt ($$n$$)**
    Pour un emprunt à annuités constantes, la $$t$$-ième part d'amortissement $$R_t$$ est donnée par la formule $$R_t = A(1+i)^{-(n-t+1)}$$ (cf. slide 97).
    Pour $$t=10$$ :
    $$R_{10} = A(1+i)^{-(n-10+1)} = A(1+i)^{-(n-9)}$$
    $$3\,984,31 = 12\,557,57(1+0,11)^{-(n-9)}$$
    $$(1,11)^{-(n-9)} = \frac{3\,984,31}{12\,557,57} \approx 0,31728500$$
    $$-(n-9)\ln(1,11) = \ln(0,31728500)$$
    $$-(n-9) \times 0,104360015... = -1,14800016...$$
    $$n-9 = \frac{-1,14800016}{-0,104360015} \approx 11,000...$$
    $$n-9 = 11 \implies n = 20$$
    La durée de l'emprunt est de $$20$$ ans.

**Réponse a) :** Le taux d'intérêt est de $$11\%$$ par an. La durée de l'emprunt est de $$20$$ ans.

**b) Quel est le solde restant dû un an avant la fin de l'emprunt ?**

Un an avant la fin de l'emprunt correspond à la fin de la période $$n-1$$. On cherche donc $$S_{n-1} = S_{19}$$.
Le solde restant dû $$S_{n-1}$$ est égal au dernier amortissement $$R_n$$, car $$S_n=0$$ et $$S_{n-1} - R_n = 0$$.
$$R_n = A(1+i)^{-(n-n+1)} = A(1+i)^{-1}$$.
$$S_{19} = R_{20} = A(1+i)^{-1} = 12\,557,57€ \times (1,11)^{-1} = \frac{12\,557,57€}{1,11} \approx 11\,313,1261€$$.

**Réponse b) :** Le solde restant dû un an avant la fin de l'emprunt ($$S_{19}$$) est de $$11\,313,13€$$.

**c) Un an avant la fin de l'emprunt, on constate une hausse de 1% du taux d'intérêt annuel, quelle sera la dernière annuité si la durée de l'emprunt reste inchangée ?**

Un an avant la fin de l'emprunt, nous sommes à la fin de la période $$t=19$$. Le solde restant dû est $$S_{19} = 11\,313,1261€$$.
Il reste une seule période de remboursement (la 20e année).
Le taux d'intérêt pour cette dernière période devient $$i' = i + 0,01 = 0,11 + 0,01 = 0,12 = 12\%$$.
La durée de l'emprunt reste $$n=20$$ ans.
La dernière annuité $$A'_{20}$$ doit couvrir le remboursement du capital $$S_{19}$$ et les intérêts $$I'_{20}$$ sur ce capital au nouveau taux $$i'$$.
$$I'_{20} = S_{19} \times i' = 11\,313,1261€ \times 0,12 = 1\,357,5751€$$.
L'amortissement pour la dernière période sera $$R'_{20} = S_{19} = 11\,313,1261€$$.
$$A'_{20} = R'_{20} + I'_{20} = S_{19}(1+i') = 11\,313,1261€ \times (1+0,12) = 11\,313,1261€ \times 1,12 \approx 12\,670,7012€$$.

**Réponse c) :** La dernière annuité sera de $$12\,670,70€$$.

**Concernant la dernière demande : "On vous demande de déterminer le montant de l'intérêt correspondant à la prime de remboursement qui a été généré chaque année."**

Le terme "prime de remboursement" n'est pas standard pour un emprunt indivis remboursable par annuités constantes. Une prime de remboursement est généralement associée aux obligations (différence entre valeur de remboursement et valeur nominale) ou à des clauses spécifiques de remboursement anticipé.
En l'absence de clarification, si la question fait référence à l'intérêt payé chaque année dans le cadre de l'emprunt initial (avant la modification de taux en c)), ce montant est $$I_t$$.
L'intérêt payé à l'année $$t$$ est $$I_t = S_{t-1} \times i$$.
Le solde $$S_{t-1}$$ peut s'exprimer comme $$A \cdot a_{n-(t-1)|i} = A \frac{1-(1+i)^{-(n-t+1)}}{i}$$.
Donc, $$I_t = i \left( A \frac{1-(1+i)^{-(n-t+1)}}{i} \right) = A(1-(1+i)^{-(n-t+1)})$$.
Avec $$A=12\,557,57€$$, $$n=20$$ ans, et $$i=11\%$$, le montant de l'intérêt généré (payé) chaque année $$t$$ est :
$$I_t = 12\,557,57€ \times (1-(1,11)^{-(20-t+1)})$$.
Par exemple :
Pour $$t=1$$, $$I_1 = 12\,557,57€ \times (1-(1,11)^{-20}) \approx 12\,557,57€ \times (1-0,124035) \approx 11\,000,00€$$. (Calculé sur $$C=100\,000€$$).
Pour $$t=10$$, $$I_{10} = 12\,557,57€ \times (1-(1,11)^{-11}) \approx 12\,557,57€ \times (1-0,317285) \approx 8\,573,26€$$, ce qui correspond à la donnée.

**Exercice 2**

Le taux de rendement à l'échéance (YTM), noté $$y$$, d'une obligation est le taux d'actualisation qui égalise la valeur actuelle de tous les flux de trésorerie futurs attendus de l'obligation (coupons et remboursement du principal) à son prix d'émission $$E$$.

**Données de l'obligation :**
*   Valeur Nominale ($$V$$) = $$10\,000€$$
*   Maturité ($$n$$) = $$5$$ ans
*   Prix d'émission ($$E$$) = $$10\,210€$$
*   Remboursement : au pair, donc Valeur de Remboursement ($$R$$) = $$V = 10\,000€$$.
*   Taux de coupon ($$j_t$$) variables :
    *   Année 1 : $$j_1 = 2,5\%$$
    *   Année 2 : $$j_2 = 3,0\%$$
    *   Année 3 : $$j_3 = 3,5\%$$
    *   Année 4 : $$j_4 = 4,0\%$$
    *   Année 5 : $$j_5 = 4,5\%$$

**Calcul des coupons annuels ($$c_t = j_t \times V$$) :**
*   $$c_1 = 0,025 \times 10\,000€ = 250€$$
*   $$c_2 = 0,030 \times 10\,000€ = 300€$$
*   $$c_3 = 0,035 \times 10\,000€ = 350€$$
*   $$c_4 = 0,040 \times 10\,000€ = 400€$$
*   $$c_5 = 0,045 \times 10\,000€ = 450€$$

**Formulation de l'équation du YTM :**
L'équation pour trouver $$y$$ est :
$$E = \frac{c_1}{(1+y)^1} + \frac{c_2}{(1+y)^2} + \frac{c_3}{(1+y)^3} + \frac{c_4}{(1+y)^4} + \frac{c_5 + R}{(1+y)^5}$$
En substituant les valeurs :
$$10\,210 = \frac{250}{(1+y)^1} + \frac{300}{(1+y)^2} + \frac{350}{(1+y)^3} + \frac{400}{(1+y)^4} + \frac{450 + 10\,000}{(1+y)^5}$$
$$10\,210 = \frac{250}{(1+y)} + \frac{300}{(1+y)^2} + \frac{350}{(1+y)^3} + \frac{400}{(1+y)^4} + \frac{10\,450}{(1+y)^5}$$

**Résolution de l'équation :**
Cette équation polynomiale en $$(1+y)$$ ne peut pas être résolue analytiquement de manière simple. Une méthode numérique (par essais successifs, solveur financier, ou interpolation) est nécessaire.

*   **Essai avec $$y = 3\% = 0,03$$ :**
    Valeur Actuelle ($$VA$$) si $$y=3\%$$ :
    $$VA(3\%) = \frac{250}{1,03} + \frac{300}{(1,03)^2} + \frac{350}{(1,03)^3} + \frac{400}{(1,03)^4} + \frac{10\,450}{(1,03)^5}$$
    $$VA(3\%) \approx 242,7184 + 282,7788 + 320,3069 + 355,3940 + 9014,2929$$
    $$VA(3\%) \approx 10\,215,49€$$
    Puisque $$VA(3\%) = 10\,215,49€ > E = 10\,210€$$, et que la valeur actuelle est une fonction décroissante du taux d'actualisation, le YTM $$y$$ doit être légèrement supérieur à $$3\%$$.

*   **Essai avec $$y = 3,02\% = 0,0302$$ :**
    $$VA(3,02\%) = \frac{250}{1,0302} + \frac{300}{(1,0302)^2} + \frac{350}{(1,0302)^3} + \frac{400}{(1,0302)^4} + \frac{10\,450}{(1,0302)^5}$$
    $$VA(3,02\%) \approx 242,6713 + 282,6371 + 320,1009 + 355,1154 + 9005,7463$$
    $$VA(3,02\%) \approx 10\,206,27€$$
    Puisque $$VA(3,02\%) = 10\,206,27€ < E = 10\,210€$$, le YTM $$y$$ est compris entre $$3\%$$ et $$3,02\%$$.

*   **Interpolation linéaire pour affiner :**
    $$y \approx y_1 + (y_2 - y_1) \frac{E - VA(y_1)}{VA(y_2) - VA(y_1)}$$
    $$y \approx 0,03 + (0,0302 - 0,03) \frac{10\,210 - 10\,215,491}{10\,206,271 - 10\,215,491}$$
    $$y \approx 0,03 + 0,0002 \times \frac{-5,491}{-9,220}$$
    $$y \approx 0,03 + 0,0002 \times 0,595553...$$
    $$y \approx 0,03 + 0,00011911...$$
    $$y \approx 0,03011911...$$

**Réponse :**
Le taux de rendement à l'échéance de cette obligation est $$y \approx 3,0119\%$$.

**Justification :**
Le taux de rendement à l'échéance (YTM) est le taux d'actualisation unique qui rend la valeur actuelle de tous les flux de trésorerie futurs de l'obligation (coupons successifs de $$250€, 300€, 350€, 400€$$, et le dernier flux comprenant le coupon de $$450€$$ plus le remboursement du principal de $$10\,000€$$) égale au prix d'émission de l'obligation ($$10\,210€$$). Ce taux est trouvé en résolvant numériquement l'équation d'actualisation ci-dessus. Les calculs itératifs montrent que ce taux est d'environ $$3,0119\%$$.

**Exercice 3**

Soit un crédit d'investissement de $$C = 50\,000€$$ contracté auprès d'une banque.
Le taux d'intérêt mensuel est $$i_m = 0,512\% = 0,00512$$.
La durée de remboursement est de $$n = 10$$ ans.
Le crédit est caractérisé par des annuités constantes payées en fin d'année.

**a) Que vaut l'annuité ?**

Le taux d'intérêt est mensuel, mais les annuités sont payées en fin d'année. Il faut d'abord calculer le taux d'intérêt annuel équivalent ($$i_a$$) au taux mensuel $$i_m$$.
$$i_a = (1 + i_m)^{12} - 1$$
$$i_a = (1 + 0,00512)^{12} - 1$$
$$i_a = (1,00512)^{12} - 1 \approx 1,0631907 - 1 \approx 0,0631907$$
Soit $$i_a \approx 6,31907\%$$.

L'annuité constante $$A$$ pour un emprunt $$C$$ sur $$n$$ années au taux annuel $$i_a$$ est (cf. slide 97) :
$$A = C \cdot \frac{i_a}{1 - (1+i_a)^{-n}}$$
$$A = 50\,000€ \cdot \frac{0,0631907}{1 - (1+0,0631907)^{-10}}$$
$$A = 50\,000€ \cdot \frac{0,0631907}{1 - (1,0631907)^{-10}}$$
$$A = 50\,000€ \cdot \frac{0,0631907}{1 - 0,5390087}$$
$$A = 50\,000€ \cdot \frac{0,0631907}{0,4609913}$$
$$A \approx 50\,000€ \cdot 0,1370759 \approx 6\,853,795€$$

**Réponse a) :** L'annuité (annuelle) est d'environ $$6\,853,80€$$.

**b) Quel est le solde restant dû après 5 ans ?**

Le solde restant dû après $$t$$ paiements ($$S_t$$) est (cf. slide 98) :
$$S_t = A \cdot \frac{1 - (1+i_a)^{-(n-t)}}{i_a}$$
Pour $$t=5$$ ans :
$$S_5 = A \cdot \frac{1 - (1+i_a)^{-(10-5)}}{i_a} = A \cdot \frac{1 - (1+i_a)^{-5}}{i_a}$$
$$S_5 = 6\,853,795€ \cdot \frac{1 - (1,0631907)^{-5}}{0,0631907}$$
$$S_5 = 6\,853,795€ \cdot \frac{1 - 0,7363239}{0,0631907}$$
$$S_5 = 6\,853,795€ \cdot \frac{0,2636761}{0,0631907}$$
$$S_5 \approx 6\,853,795€ \cdot 4,17274 \approx 28\,594,92€$$

**Réponse b) :** Le solde restant dû après 5 ans est d'environ $$28\,594,92€$$.

**c) Il s'est passé 5 ans depuis le début de l'emprunt et on constate une hausse de 1% du taux annuel, quelle sera la nouvelle annuité si la durée du crédit reste inchangée ?**

Après 5 ans, le solde restant dû est $$S_5 = 28\,594,92€$$.
La durée restante du crédit est $$n-t = 10 - 5 = 5$$ ans.
Le taux annuel initial était $$i_a \approx 6,31907\%$$.
Le nouveau taux annuel $$i'_a$$ est $$i_a + 0,01 = 0,0631907 + 0,01 = 0,0731907 = 7,31907\%$$.
La durée restante reste $$n' = 5$$ ans.
La nouvelle annuité $$A'$$ se calcule sur le solde $$S_5$$ avec le nouveau taux $$i'_a$$ pour les $$n'$$ périodes restantes (cf. slide 105) :
$$A' = S_5 \cdot \frac{i'_a}{1 - (1+i'_a)^{-n'}}$$
$$A' = 28\,594,92€ \cdot \frac{0,0731907}{1 - (1+0,0731907)^{-5}}$$
$$A' = 28\,594,92€ \cdot \frac{0,0731907}{1 - (1,0731907)^{-5}}$$
$$A' = 28\,594,92€ \cdot \frac{0,0731907}{1 - 0,7025896}$$
$$A' = 28\,594,92€ \cdot \frac{0,0731907}{0,2974104}$$
$$A' \approx 28\,594,92€ \cdot 0,246092 \approx 7\,037,00€$$

**Réponse c) :** La nouvelle annuité sera d'environ $$7\,037,00€$$.

**d) Que valent l'annuité et le solde restant dû après 4 ans si les annuités sont payées en début d'année ? On garde le même taux annuel que celui des sous-questions (a) et (b).**

Le taux annuel est $$i_a \approx 6,31907\%$$. Le capital emprunté est $$C=50\,000€$$. La durée est $$n=10$$ ans.
Si les annuités $$A_{deb}$$ sont payées en début d'année, l'équation d'équivalence devient :
$$C = A_{deb} + A_{deb} \cdot a_{n-1|i_a} = A_{deb} (1 + a_{n-1|i_a})$$
ou, de manière équivalente, $$C = A_{deb} \cdot \ddot{a}_{n|i_a} = A_{deb} \cdot a_{n|i_a} \cdot (1+i_a)$$.
Donc $$A_{deb} = \frac{A_{fin}}{1+i_a}$$, où $$A_{fin}$$ est l'annuité de fin de période calculée en a).
$$A_{deb} = \frac{6\,853,795€}{1,0631907} \approx 6\,446,29€$$.

**Annuité :** L'annuité payée en début d'année est d'environ $$6\,446,29€$$.

**Solde restant dû après 4 ans (i.e. après le paiement de la 4ème annuité de début de période) :**
Lorsqu'une annuité est payée en début de période $$t$$, elle est équivalente à une annuité payée en fin de période $$t-1$$ en termes d'actualisation.
Le solde restant dû après le paiement de la $$t$$-ième annuité de début de période, noté $$S_{deb, t}$$, est le même que le solde restant dû après le paiement de la $$t$$-ième annuité de fin de période pour un emprunt qui commencerait un an plus tard, ou de manière plus directe, c'est la valeur actuelle des $$(n-t)$$ annuités de début de période restantes.
$$S_{deb,t} = A_{deb} \cdot \ddot{a}_{n-t|i_a} = A_{deb} \cdot \frac{1-(1+i_a)^{-(n-t)}}{i_a} (1+i_a)$$
Pour $$t=4$$ :
$$S_{deb,4} = A_{deb} \cdot \frac{1-(1+i_a)^{-(10-4)}}{i_a} (1+i_a) = A_{deb} \cdot \frac{1-(1+i_a)^{-6}}{i_a} (1+i_a)$$
$$S_{deb,4} = 6\,446,29€ \cdot \frac{1-(1,0631907)^{-6}}{0,0631907} \cdot (1,0631907)$$
$$S_{deb,4} = 6\,446,29€ \cdot \frac{1-0,692595}{0,0631907} \cdot (1,0631907)$$
$$S_{deb,4} = 6\,446,29€ \cdot \frac{0,307405}{0,0631907} \cdot (1,0631907)$$
$$S_{deb,4} \approx 6\,446,29€ \cdot 4,86485 \cdot (1,0631907) \approx 33\,299,69€$$.

Alternativement, le solde après $$t$$ paiements de début est $$S_{deb,t}$$.
Le premier paiement $$A_{deb}$$ est à $$t=0$$. Le capital restant après ce premier paiement est $$C - A_{deb}$$.
Ce capital $$C' = C - A_{deb}$$ est ensuite remboursé par $$n-1$$ annuités de fin de période $$A_{deb}$$.
Le solde après 4 paiements de début est donc le solde après 3 paiements de fin de période $$A_{deb}$$ sur le capital $$C'$$.
$$C' = 50\,000€ - 6\,446,29€ = 43\,553,71€$$.
Nombre d'annuités restantes pour $$C'$$ : $$n-1=9$$.
Solde après 3 annuités $$A_{deb}$$ sur $$C'$$ (ce qui correspond à 4 paiements en début de période sur $$C$$) :
$$S'_{3} = A_{deb} \cdot \frac{1 - (1+i_a)^{-((n-1)-3)}}{i_a} = A_{deb} \cdot \frac{1 - (1+i_a)^{-(9-3)}}{i_a} = A_{deb} \cdot \frac{1 - (1+i_a)^{-6}}{i_a}$$
$$S'_{3} = 6\,446,29€ \cdot \frac{1-(1,0631907)^{-6}}{0,0631907} \approx 6\,446,29€ \cdot 4,86485 \approx 31\,353,40€$$.
Ce calcul correspond au solde *juste avant* le paiement de la 5ème annuité de début de période.
Si la question "solde restant dû après 4 ans" signifie *après le paiement de la 4ème annuité de début de période*, alors le calcul précédent de $$31\,353,40€$$ est le solde *avant* le 4ème paiement de $$A_{deb}$$.
Le solde *après* le 4ème paiement de $$A_{deb}$$ est $$S'_{3} - A_{deb}$$ si on considère $$t=0$$ comme le premier paiement.
Non, la convention est que $$S_t$$ est le solde *après* le paiement à l'instant $$t$$.
Si les paiements sont en début de période $$k=0, 1, ..., n-1$$.
Le solde $$S_{deb,k}$$ après le paiement à l'instant $$k$$ est la valeur actuelle des paiements futurs $$A_{deb}$$ aux instants $$k+1, ..., n-1$$, actualisés à l'instant $$k$$.
$$S_{deb,k} = A_{deb} \sum_{j=1}^{n-1-k} (1+i_a)^{-j} = A_{deb} \cdot a_{n-1-k|i_a}$$.
Après 4 ans, cela signifie après le paiement à l'instant $$k=3$$ (car les instants sont $$0,1,2,3,...$$).
$$S_{deb,3} = A_{deb} \cdot a_{10-1-3|i_a} = A_{deb} \cdot a_{6|i_a}$$.
$$a_{6|i_a} = \frac{1-(1,0631907)^{-6}}{0,0631907} \approx 4,86485$$.
$$S_{deb,3} = 6\,446,29€ \cdot 4,86485 \approx 31\,353,40€$$.

**Réponse d) :**
L'annuité payée en début d'année est d'environ $$6\,446,29€$$.
Le solde restant dû après 4 ans (c'est-à-dire après le paiement de la 4ème annuité qui a lieu au début de la 4ème année, soit à $$t=3$$) est d'environ $$31\,353,40€$$.

**Exercice 4**

On considère une obligation à terme fixe d'échéance $$n=10$$ ans, de valeur nominale $$V=100€$$.

**a) Si le taux nominal est de 4,5% et que l'obligation est émise et remboursée au pair, que vaut le taux de rendement à l'échéance ?**

*   Taux nominal (taux de coupon) $$j = 4,5\% = 0,045$$.
*   Le coupon annuel $$c = j \times V = 0,045 \times 100€ = 4,5€$$.
*   Émise au pair signifie que le prix d'émission $$E = V = 100€$$.
*   Remboursée au pair signifie que la valeur de remboursement $$R = V = 100€$$.

Lorsque l'obligation est émise au pair ($$E=V$$) et remboursée au pair ($$R=V$$), et que les coupons sont constants, le taux de rendement à l'échéance $$y$$ est égal au taux de coupon $$j$$. (cf. slide 116, cas 1 : si $$y=j$$, $$E=V$$).
Ici, $$E=V=100€$$, $$R=V=100€$$.
L'équation du prix d'émission est (cf. slide 114) :
$$E = c \cdot \frac{1-(1+y)^{-n}}{y} + \frac{R}{(1+y)^n}$$
Si $$y=j=0,045$$, alors :
$$E = 4,5 \cdot \frac{1-(1+0,045)^{-10}}{0,045} + \frac{100}{(1+0,045)^{10}}$$
$$E = 4,5 \cdot a_{10|0,045} + 100 \cdot (1,045)^{-10}$$
$$a_{10|0,045} = \frac{1-(1,045)^{-10}}{0,045} \approx \frac{1-0,64392768}{0,045} \approx \frac{0,35607232}{0,045} \approx 7,912718$$
$$E \approx 4,5 \times 7,912718 + 100 \times 0,64392768$$
$$E \approx 35,607231 + 64,392768 = 100€$$.
Comme $$E=100€$$ pour $$y=0,045$$, le taux de rendement à l'échéance est bien $$4,5\%$$.

**Réponse a) :** Le taux de rendement à l'échéance est de $$4,5\%$$.

**b) Si on désire garantir un taux de rendement à l'échéance de 5% en gardant le taux nominal de 4,5% et un remboursement au pair, que doit valoir le prix d'émission ?**

*   Taux de rendement à l'échéance désiré $$y = 5\% = 0,05$$.
*   Taux nominal $$j = 4,5\% \implies c = 4,5€$$.
*   Remboursement au pair $$R = V = 100€$$.
*   Maturité $$n=10$$ ans.

On cherche le prix d'émission $$E$$:
$$E = c \cdot \frac{1-(1+y)^{-n}}{y} + \frac{R}{(1+y)^n}$$
$$E = 4,5 \cdot \frac{1-(1+0,05)^{-10}}{0,05} + \frac{100}{(1+0,05)^{10}}$$
$$E = 4,5 \cdot a_{10|0,05} + 100 \cdot (1,05)^{-10}$$
$$a_{10|0,05} = \frac{1-(1,05)^{-10}}{0,05} \approx \frac{1-0,61391325}{0,05} \approx \frac{0,38608675}{0,05} \approx 7,721735$$
$$E \approx 4,5 \times 7,721735 + 100 \times 0,61391325$$
$$E \approx 34,7478075 + 61,391325 \approx 96,1391325€$$.

**Réponse b) :** Le prix d'émission doit valoir environ $$96,14€$$.

**c) Si le taux nominal est de 4% et que le prix d'émission est celui calculé à la sous-question précédente, quelle doit être la valeur de remboursement de l'obligation pour avoir un taux de rendement à l'échéance de 5% ?**

*   Nouveau taux nominal $$j' = 4\% = 0,04 \implies c' = j' \times V = 0,04 \times 100€ = 4€$$.
*   Prix d'émission $$E \approx 96,1391325€$$ (calculé en b)).
*   Taux de rendement à l'échéance désiré $$y = 5\% = 0,05$$.
*   Maturité $$n=10$$ ans.
On cherche la valeur de remboursement $$R'$$.
$$E = c' \cdot \frac{1-(1+y)^{-n}}{y} + \frac{R'}{(1+y)^n}$$
$$96,1391325 = 4 \cdot a_{10|0,05} + \frac{R'}{(1,05)^{10}}$$
$$a_{10|0,05} \approx 7,721735$$
$$(1,05)^{-10} \approx 0,61391325$$
$$96,1391325 = 4 \times 7,721735 + R' \times 0,61391325$$
$$96,1391325 = 30,88694 + 0,61391325 \cdot R'$$
$$96,1391325 - 30,88694 = 0,61391325 \cdot R'$$
$$65,2521925 = 0,61391325 \cdot R'$$
$$R' = \frac{65,2521925}{0,61391325} \approx 106,2891€$$.

**Réponse c) :** La valeur de remboursement de l'obligation doit être d'environ $$106,29€$$.

**d) Si l'obligation est une obligation zéro-coupon et que le remboursement se fait au pair, quel est le prix d'émission de l'obligation si le taux de rendement à l'échéance est de 5% ?**

*   Obligation zéro-coupon : aucun coupon périodique ($$c=0$$).
*   Remboursement au pair : $$R = V = 100€$$.
*   Taux de rendement à l'échéance $$y = 5\% = 0,05$$.
*   Maturité $$n=10$$ ans.
Le prix d'émission $$E$$ d'une obligation zéro-coupon est (cf. slide 132) :
$$E = \frac{R}{(1+y)^n}$$
$$E = \frac{100}{(1+0,05)^{10}}$$
$$E = \frac{100}{(1,05)^{10}} \approx \frac{100}{1,6288946} \approx 100 \times 0,61391325 \approx 61,391325€$$.

**Réponse d) :** Le prix d'émission de l'obligation zéro-coupon est d'environ $$61,39€$$.

**Exercice 5 : LE SEUL OU IL A EU MAUVAIS**

Les flux de trésorerie sont donnés en fin d'année (en k€).
Le coût d'opportunité du capital (taux d'actualisation) est $$k = 10\% = 0,10$$.

**a) Déterminer $$x$$ et $$y$$ et de calculer la VAN de chaque projet, en considérant que le coût d'opportunité du capital est de 10%.**

*   **Détermination de $$x$$ (Flux année 0 du Projet B)**
    Le TRI d'un projet est le taux d'actualisation pour lequel sa VAN est nulle.
    Pour le Projet B, TRI = $$55\% = 0,55$$. Flux : $$F_0 = x$$, $$F_1 = 0$$, $$F_2 = 206$$, $$F_3 = 95$$.
    $$VAN_B(TRI_B) = x + \frac{0}{(1+0,55)^1} + \frac{206}{(1+0,55)^2} + \frac{95}{(1+0,55)^3} = 0$$
    $$x + 0 + \frac{206}{(1,55)^2} + \frac{95}{(1,55)^3} = 0$$
    $$x + \frac{206}{2,4025} + \frac{95}{3,723875} = 0$$
    $$x + 85,7439... + 25,5109... = 0$$
    $$x + 111,2548... = 0 \implies x \approx -111,25$$ k€

*   **Détermination de $$y$$ (partie du Flux année 3 du Projet C)**
    Pour le Projet C, TRI = $$50\% = 0,50$$. Flux : $$F_0 = -100$$, $$F_1 = 37$$, $$F_2 = 0$$, $$F_3 = 204+y$$.
    $$VAN_C(TRI_C) = -100 + \frac{37}{(1+0,50)^1} + \frac{0}{(1+0,50)^2} + \frac{204+y}{(1+0,50)^3} = 0$$
    $$-100 + \frac{37}{1,50} + 0 + \frac{204+y}{(1,50)^3} = 0$$
    $$-100 + 24,6666... + \frac{204+y}{3,375} = 0$$
    $$-75,3333... + \frac{204+y}{3,375} = 0$$
    $$\frac{204+y}{3,375} = 75,3333...$$
    $$204+y = 75,3333... \times 3,375 = 254,25$$
    $$y = 254,25 - 204 = 50,25$$ k€

*   **Calcul de la VAN de chaque projet avec $$k=10\%$$**
    **Projet A :** $$F_0 = -100$$, $$F_1 = 30$$, $$F_2 = 153$$, $$F_3 = 88$$.
    $$VAN_A = -100 + \frac{30}{(1,1)^1} + \frac{153}{(1,1)^2} + \frac{88}{(1,1)^3}$$
    $$VAN_A = -100 + \frac{30}{1,1} + \frac{153}{1,21} + \frac{88}{1,331}$$
    $$VAN_A = -100 + 27,2727... + 126,4462... + 66,0405...$$
    $$VAN_A \approx -100 + 219,7594 \approx 119,76$$ k€

    **Projet B :** $$F_0 = -111,2548$$, $$F_1 = 0$$, $$F_2 = 206$$, $$F_3 = 95$$.
    $$VAN_B = -111,2548 + \frac{0}{(1,1)^1} + \frac{206}{(1,1)^2} + \frac{95}{(1,1)^3}$$
    $$VAN_B = -111,2548 + 0 + \frac{206}{1,21} + \frac{95}{1,331}$$
    $$VAN_B = -111,2548 + 170,2479... + 71,3749...$$
    $$VAN_B \approx -111,2548 + 241,6228 \approx 130,37$$ k€

    **Projet C :** $$F_0 = -100$$, $$F_1 = 37$$, $$F_2 = 0$$, $$F_3 = 204+50,25 = 254,25$$.
    $$VAN_C = -100 + \frac{37}{(1,1)^1} + \frac{0}{(1,1)^2} + \frac{254,25}{(1,1)^3}$$
    $$VAN_C = -100 + \frac{37}{1,1} + 0 + \frac{254,25}{1,331}$$
    $$VAN_C = -100 + 33,6363... + 190,9466...$$
    $$VAN_C \approx -100 + 224,5829 \approx 124,58$$ k€

**Réponse a) :**
$$x \approx -111,25$$ k€
$$y = 50,25$$ k€
$$VAN_A \approx 119,76$$ k€
$$VAN_B \approx 130,37$$ k€
$$VAN_C \approx 124,58$$ k€

**b) Déterminer le projet le plus intéressant sur base de la VAN, de justifier et d'interpréter le résultat obtenu.**

Sur la base de la Valeur Actuelle Nette (VAN) calculée avec un coût d'opportunité du capital de 10% :
*   $$VAN_A \approx 119,76$$ k€
*   $$VAN_B \approx 130,37$$ k€
*   $$VAN_C \approx 124,58$$ k€

Le projet B a la VAN la plus élevée ($$130,37$$ k€).
La règle de la VAN stipule qu'il faut choisir le projet ayant la VAN la plus élevée (cf. slide 41).
**Justification :** La VAN représente la richesse créée par le projet, actualisée à la date d'aujourd'hui, au-delà du rendement exigé par le coût d'opportunité du capital. Un projet avec une VAN positive est acceptable car il crée de la valeur. Entre plusieurs projets mutuellement exclusifs, celui qui crée le plus de valeur est le meilleur.
**Interprétation :** Le projet B est celui qui devrait générer le plus de richesse pour l'entreprise, une fois que les flux futurs sont actualisés au coût du capital de 10%. Il est estimé que le projet B rapportera l'équivalent de $$130\,370€$$ en valeur d'aujourd'hui, après avoir couvert l'investissement initial et le coût de financement de cet investissement.

**Réponse b) :** Le projet le plus intéressant sur la base de la VAN est le Projet B, avec une $$VAN_B \approx 130,37$$ k€. Ce projet est celui qui est attendu pour créer le plus de valeur actuelle pour l'entreprise.

**c) D'expliquer pourquoi la règle du TRI n'est pas pertinente dans le cas présent.**

Les TRI donnés sont :
*   Projet A : $$60\%$$
*   Projet B : $$55\%$$
*   Projet C : $$50\%$$

Si on se basait uniquement sur le TRI, le Projet A semblerait le plus intéressant. Cependant, le Projet B a la VAN la plus élevée.

La règle du TRI peut être source d'erreurs, notamment quand (cf. slide 41) :
1.  **Les échelles des projets sont différentes :** Ici, les investissements initiaux ($$F_0$$) sont de $$-100$$ k€ pour A et C, mais de $$-111,25$$ k€ pour B. Les projets ne sont donc pas de même échelle d'investissement. Un TRI élevé sur un petit investissement peut générer moins de VAN qu'un TRI plus faible sur un investissement plus important.
2.  **Les calendriers des flux de trésorerie sont différents :**
    *   Projet A : Flux importants en années 2 et 3.
    *   Projet B : Pas de flux en année 1, flux importants en années 2 et 3.
    *   Projet C : Flux en année 1, pas de flux en année 2, flux important en année 3.
    Les profils temporels des flux diffèrent. Le TRI ne tient pas compte de la manière dont les flux sont réinvestis (il suppose implicitement un réinvestissement au TRI lui-même, ce qui n'est pas toujours réaliste), alors que la VAN suppose un réinvestissement au coût du capital.

Dans ce cas, l'échelle d'investissement du projet B est légèrement supérieure à celle des projets A et C. Même si son TRI est inférieur à celui du projet A, la valeur absolue de la richesse créée (VAN) par le projet B est supérieure.
La VAN est généralement considérée comme le critère de décision supérieur car elle mesure directement la création de valeur en unités monétaires actuelles et utilise un taux de réinvestissement plus réaliste (le coût du capital). Le TRI est un taux relatif et ne reflète pas la magnitude de la valeur créée.

**Réponse c) :** La règle du TRI n'est pas pertinente dans le cas présent principalement parce que les projets ont des échelles d'investissement initial différentes (Projet B a un $$F_0$$ de $$-111,25$$ k€ contre $$-100$$ k€ pour A et C). De plus, les calendriers des flux de trésorerie diffèrent, ce qui peut aussi affecter la pertinence du TRI par rapport à la VAN. La VAN est un meilleur indicateur de la création de valeur absolue.

**Exercice 6**

On considère quatre obligations A, B, C, D dont la valeur nominale est $$V=100€$$ et qui sont remboursées au pair ($$R=V=100€$$).
Le taux de rendement à l'échéance initial est $$y_{initial} = 6\% = 0,06$$.
On nous demande la variation du prix (en pourcentage, au centième près) de ces obligations si le taux de rendement à l'échéance passe à $$y_{nouveau} = 5\% = 0,05$$.

**Formule du prix d'une obligation à terme fixe (cf. slide 114) :**
$$P = c \cdot a_{n|y} + R \cdot (1+y)^{-n} = c \cdot \frac{1-(1+y)^{-n}}{y} + R \cdot (1+y)^{-n}$$
Pour les obligations zéro-coupon (A et B), $$c=0$$, donc $$P = R \cdot (1+y)^{-n}$$.

**Calcul du prix initial ($$P_{initial}$$) pour chaque obligation avec $$y_{initial} = 6\%$$ :**

*   **Obligation A :** Taux de coupon $$j_A=0\% \implies c_A=0$$. Maturité $$n_A=15$$ ans.
    $$P_{A,initial} = 100 \cdot (1+0,06)^{-15} = 100 \cdot (1,06)^{-15} \approx 100 \cdot 0,41726506 \approx 41,7265€$$.

*   **Obligation B :** Taux de coupon $$j_B=0\% \implies c_B=0$$. Maturité $$n_B=10$$ ans.
    $$P_{B,initial} = 100 \cdot (1+0,06)^{-10} = 100 \cdot (1,06)^{-10} \approx 100 \cdot 0,55839478 \approx 55,8395€$$.

*   **Obligation C :** Taux de coupon $$j_C=4\% \implies c_C = 0,04 \times 100 = 4€$$. Maturité $$n_C=15$$ ans.
    $$a_{15|0,06} = \frac{1-(1,06)^{-15}}{0,06} \approx \frac{1-0,41726506}{0,06} \approx \frac{0,58273494}{0,06} \approx 9,712249$$.
    $$P_{C,initial} = 4 \cdot a_{15|0,06} + 100 \cdot (1,06)^{-15} \approx 4 \cdot 9,712249 + 100 \cdot 0,41726506$$
    $$P_{C,initial} \approx 38,848996 + 41,726506 \approx 80,5755€$$.

*   **Obligation D :** Taux de coupon $$j_D=8\% \implies c_D = 0,08 \times 100 = 8€$$. Maturité $$n_D=10$$ ans.
    $$a_{10|0,06} = \frac{1-(1,06)^{-10}}{0,06} \approx \frac{1-0,55839478}{0,06} \approx \frac{0,44160522}{0,06} \approx 7,360087$$.
    $$P_{D,initial} = 8 \cdot a_{10|0,06} + 100 \cdot (1,06)^{-10} \approx 8 \cdot 7,360087 + 100 \cdot 0,55839478$$
    $$P_{D,initial} \approx 58,880696 + 55,839478 \approx 114,7202€$$.

**Calcul du nouveau prix ($$P_{nouveau}$$) pour chaque obligation avec $$y_{nouveau} = 5\%$$ :**

*   **Obligation A :** $$c_A=0$$, $$n_A=15$$ ans.
    $$P_{A,nouveau} = 100 \cdot (1+0,05)^{-15} = 100 \cdot (1,05)^{-15} \approx 100 \cdot 0,48101711 \approx 48,1017€$$.

*   **Obligation B :** $$c_B=0$$, $$n_B=10$$ ans.
    $$P_{B,nouveau} = 100 \cdot (1+0,05)^{-10} = 100 \cdot (1,05)^{-10} \approx 100 \cdot 0,61391325 \approx 61,3913€$$.

*   **Obligation C :** $$c_C=4€$$, $$n_C=15$$ ans.
    $$a_{15|0,05} = \frac{1-(1,05)^{-15}}{0,05} \approx \frac{1-0,48101711}{0,05} \approx \frac{0,51898289}{0,05} \approx 10,379658$$.
    $$P_{C,nouveau} = 4 \cdot a_{15|0,05} + 100 \cdot (1,05)^{-15} \approx 4 \cdot 10,379658 + 100 \cdot 0,48101711$$
    $$P_{C,nouveau} \approx 41,518632 + 48,101711 \approx 89,6203€$$.

*   **Obligation D :** $$c_D=8€$$, $$n_D=10$$ ans.
    $$a_{10|0,05} = \frac{1-(1,05)^{-10}}{0,05} \approx \frac{1-0,61391325}{0,05} \approx \frac{0,38608675}{0,05} \approx 7,721735$$.
    $$P_{D,nouveau} = 8 \cdot a_{10|0,05} + 100 \cdot (1,05)^{-10} \approx 8 \cdot 7,721735 + 100 \cdot 0,61391325$$
    $$P_{D,nouveau} \approx 61,77388 + 61,391325 \approx 123,1652€$$.

**Calcul de la variation du prix en pourcentage :**
Variation ($$\%$$) = $$\frac{P_{nouveau} - P_{initial}}{P_{initial}} \times 100\%$$

*   **Obligation A :**
    Variation$$_A = \frac{48,1017 - 41,7265}{41,7265} \times 100\% = \frac{6,3752}{41,7265} \times 100\% \approx 0,15278 \times 100\% \approx 15,28\%$$.

*   **Obligation B :**
    Variation$$_B = \frac{61,3913 - 55,8395}{55,8395} \times 100\% = \frac{5,5518}{55,8395} \times 100\% \approx 0,099424 \times 100\% \approx 9,94\%$$.

*   **Obligation C :**
    Variation$$_C = \frac{89,6203 - 80,5755}{80,5755} \times 100\% = \frac{9,0448}{80,5755} \times 100\% \approx 0,11225 \times 100\% \approx 11,23\%$$.

*   **Obligation D :**
    Variation$$_D = \frac{123,1652 - 114,7202}{114,7202} \times 100\% = \frac{8,4450}{114,7202} \times 100\% \approx 0,073615 \times 100\% \approx 7,36\%$$.

**Réponse :**
La variation du prix (en pourcentage, au centième près) de ces obligations si le taux de rendement à l'échéance passe de 6% à 5% est :
*   Obligation A : $$15,28\%$$
*   Obligation B : $$9,94\%$$
*   Obligation C : $$11,23\%$$
*   Obligation D : $$7,36\%$$

**Exercice 7**

Une entreprise anticipe pour la fin de l'année un bénéfice par action (BPA) de $$BPA_1 = 5€$$.
Elle envisage de verser un dividende de $$D_1 = 3€$$ par action.
Le bénéfice non distribué sera investi dans de nouveaux projets ayant une rentabilité estimée de $$R_{np} = 15\%$$ par an.
Le taux de distribution des dividendes et la rentabilité des nouveaux investissements restent constants par la suite.
Le coût d'opportunité du capital (taux de rentabilité exigé par les actionnaires) est $$y = 12\%$$.

**a) Déterminer le taux de croissance des bénéfices de l'entreprise.**

Bénéfice par action à la fin de l'année 1 : $$BPA_1 = 5€$$.
Dividende versé par action à la fin de l'année 1 : $$D_1 = 3€$$.
Bénéfice non distribué par action (réinvesti) : $$BND_1 = BPA_1 - D_1 = 5€ - 3€ = 2€$$.
Le taux de rétention des bénéfices ($$b$$) est la part du bénéfice qui est réinvestie.
$$b = \frac{BND_1}{BPA_1} = \frac{2€}{5€} = 0,4$$.
(Le taux de distribution des dividendes $$d = 1-b = \frac{D_1}{BPA_1} = \frac{3}{5} = 0,6$$).

Le taux de croissance des bénéfices ($$g$$) est donné par la formule (cf. slide 150) :
$$g = \text{Taux de rétention des bénéfices} \times \text{Rentabilité des nouveaux investissements}$$
$$g = b \times R_{np}$$
$$g = 0,4 \times 15\% = 0,4 \times 0,15 = 0,06$$.
Soit $$g = 6\%$$.

**Réponse a) :** Le taux de croissance des bénéfices de l'entreprise est de $$6\%$$.

**b) Calculer le prix d'une action de l'entreprise aujourd'hui.**

On utilise le modèle de Gordon-Shapiro (dividendes croissants à taux constant, cf. slide 147), car le taux de distribution est constant, ce qui implique que les dividendes croîtront au même taux que les bénéfices, soit $$g=6\%$$.
Le taux de rentabilité exigé par les actionnaires est $$y = 12\%$$.
Le dividende attendu à la fin de la première année est $$D_1 = 3€$$.
Le prix de l'action aujourd'hui ($$P_0$$) est :
$$P_0 = \frac{D_1}{y-g}$$
Il faut que $$y > g$$, ce qui est le cas ici ($$12\% > 6\%$$).
$$P_0 = \frac{3€}{0,12 - 0,06} = \frac{3€}{0,06} = 50€$$.

**Réponse b) :** Le prix d'une action de l'entreprise aujourd'hui est de $$50€$$.

**c) Calculer le prix d'une action de l'entreprise aujourd'hui si finalement, le dividende versé à la fin de l'année est de 4€ et que le taux de distribution des dividendes reste constant par la suite.**

Nouveau dividende versé à la fin de l'année 1 : $$D'_1 = 4€$$.
Le bénéfice par action à la fin de l'année 1 reste $$BPA_1 = 5€$$.
Nouveau taux de distribution des dividendes $$d' = \frac{D'_1}{BPA_1} = \frac{4€}{5€} = 0,8$$.
Le taux de distribution reste constant par la suite à $$d'=0,8$$.
Nouveau taux de rétention des bénéfices $$b' = 1 - d' = 1 - 0,8 = 0,2$$.
La rentabilité des nouveaux investissements reste $$R_{np} = 15\%$$.
Nouveau taux de croissance des bénéfices (et des dividendes) $$g'$$ :
$$g' = b' \times R_{np} = 0,2 \times 0,15 = 0,03$$.
Soit $$g' = 3\%$$.
Le taux de rentabilité exigé par les actionnaires reste $$y = 12\%$$.
Il faut que $$y > g'$$, ce qui est le cas ($$12\% > 3\%$$).
Nouveau prix de l'action aujourd'hui ($$P'_0$$) :
$$P'_0 = \frac{D'_1}{y-g'}$$
$$P'_0 = \frac{4€}{0,12 - 0,03} = \frac{4€}{0,09} \approx 44,444...€$$.

**Réponse c) :** Si le dividende versé à la fin de l'année est de $$4€$$ et que ce nouveau taux de distribution reste constant, le prix d'une action de l'entreprise aujourd'hui serait d'environ $$44,44€$$.

**Exercice 8**

Il existe une action $$i$$ côtée sur le marché d'Euronext Paris offrant un dividende constant dans le temps de $$D = 2,1€$$.
La rentabilité attendue de l'action $$y_i$$ est celle du MEDAF (ou CAPM).

Données :
*   Rendement annuel du taux sans risque : $$R_F = 2,33\% = 0,0233$$.
*   Espérance de rendement annuel du CAC 40 (portefeuille de marché) : $$E(R_M) = 5,88\% = 0,0588$$.
*   Beta entre le titre $$i$$ et le portefeuille de marché : $$\beta_i = 0,35$$.

**Calcul de la rentabilité attendue de l'action $$y_i$$ selon le MEDAF (cf. slide 63) :**
$$y_i = R_F + \beta_i (E(R_M) - R_F)$$
$$y_i = 0,0233 + 0,35 \times (0,0588 - 0,0233)$$
$$y_i = 0,0233 + 0,35 \times 0,0355$$
$$y_i = 0,0233 + 0,012425$$
$$y_i = 0,035725 = 3,5725\%$$.

**Calcul de la valeur de cette action au temps 0 ($$P_0$$) :**
Puisque le dividende est constant ($$D = 2,1€$$), il s'agit d'une rente perpétuelle. La valeur de l'action est l'actualisation de cette rente perpétuelle au taux $$y_i$$.
$$P_0 = \frac{D}{y_i}$$
$$P_0 = \frac{2,1€}{0,035725} \approx 58,782365€$$.

La valeur de cette action au temps 0 est d'environ $$58,78€$$.

**Deuxième partie de la question :**
L'action est actuellement valorisée à $$P'_{0,marché} = 72,03€$$ sur le marché.
On considère maintenant que le dividende n'est plus constant mais va croître selon un taux de croissance $$g$$.
On utilise le taux de rentabilité attendue $$y_i = 0,035725$$ calculé précédemment.
Le dividende à la fin de la première période est $$D_1$$. On suppose que le dividende de $$2,1€$$ est le dividende qui vient d'être versé ($$D_0$$) ou le prochain dividende attendu si la croissance n'avait pas encore été prise en compte. Si on considère que $$D_1$$ est le premier dividende affecté par la croissance, et que le "dividende constant de 2,1€" est une base $$D_0$$, alors $$D_1 = D_0(1+g) = 2,1(1+g)$$.
Cependant, il est plus courant, si le dividende *offert* est de 2,1€ et qu'une croissance est ensuite introduite, de considérer $$D_1 = 2,1€$$ comme le premier dividende d'une série croissante. Adoptons cette interprétation.

Si $$D_1 = 2,1€$$ est le premier dividende d'une série croissante, et que la valeur de marché est $$P'_{0,marché} = 72,03€$$, on utilise le modèle de Gordon-Shapiro :
$$P'_{0,marché} = \frac{D_1}{y_i - g}$$
$$72,03 = \frac{2,1}{0,035725 - g}$$
$$0,035725 - g = \frac{2,1}{72,03}$$
$$0,035725 - g \approx 0,029154519$$
$$g \approx 0,035725 - 0,029154519$$
$$g \approx 0,006570481$$.

Soit $$g \approx 0,657\%$$.

Pour que le modèle de Gordon-Shapiro soit valide, il faut $$y_i > g$$.
$$0,035725 > 0,006570481$$, ce qui est bien le cas.

**Réponse :**
En supposant que $$y_i$$ est la rentabilité attendue de l'action (calculée par le MEDAF à $$3,5725\%$$) et que le dividende de $$2,1€$$ est le premier dividende $$D_1$$ d'une série croissante, pour que l'action soit valorisée à $$72,03€$$ sur le marché, le taux de croissance $$g$$ des dividendes doit valoir environ $$0,657\%$$.
