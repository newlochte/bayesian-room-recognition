# Scenariusz filmu — Semantyczna Detekcja Pomieszczeń

Format kolumn: **[NARRACJA]** | **[ANIMACJA]**

---

## SCENA 1 — Wstęp / Problem

**[NARRACJA]**
Wyobraź sobie, że wchodzisz do pokoju. Widzisz lodówkę, zlew, piekarnik. W ułamku sekundy wiesz, że to kuchnia. Nie myślisz o tym świadomie.

**[ANIMACJA]**
Czarne tło. Pojawia się zdjęcie kuchni. Na kolejnych obiektach (lodówka, zlew, kuchenka) pojawiają się kolorowe ramki z etykietami. W centrum ekranu pojawia się napis: `Kuchnia`. Animacja — obiekty "wlatują" w etykietę pomieszczenia.

---

**[NARRACJA]**
Dla robota to nie jest takie proste. Robot widzi piksele. Żeby zrozumiał polecenie "przynieś mi jogurt z kuchni", musi najpierw wiedzieć, gdzie ta kuchnia jest. To właśnie jest problem semantycznej detekcji pomieszczeń.

**[ANIMACJA]**
Animacja robota z kamerą. Z kamery wychodzi strumień pikseli / siatka liczb. Następnie pojawia się pytanie: `Gdzie jest kuchnia?` Obok mapa budynku z zaznaczonymi pokojami.

---

## SCENA 2 — Przegląd Architektury (ogólna idea)

**[NARRACJA]**
Nasz pomysł jest prosty: nie analizujemy pojedyńczych pikseli. Patrzymy co znajduje się na zdjęciu - wykrywamy obiekty. Następnie na ich podstawie wnioskujemy, jaki to typ pomieszczenia.

**[ANIMACJA]**
Schemat potoku dwóch bloków:

```
[Zdjęcie] → [YOLO: detekcja obiektów] → [Sieć bayesowska: klasyfikacja pomieszczenia]
```

Bloki pojawiają się kolejno, strzałki "wlatują" między nimi.

---

## SCENA 3 — Krok 1: Detekcja obiektów z YOLO

**[NARRACJA]**
Weźmy zdjęcie biura. Model YOLO wykrył 6 obiektów. Każdy z tych obiektów ma swoją pewność. Odcinamy obiekty których model nie był dość pewny. Obecność tych obiektów staje się wejściem naszej sieci.


**[ANIMACJA]**
Przykładowe zdjęcie pomieszczenia. Na nim pojawiają się bounding boxy z etykietami. Boxy przemiesczają się na środek. Pojawiaja się ich prawdopodobieństwa. Następuje odcięcie najgorszych

---

**[NARRACJA]**
Sieć bayesa ma tylko jeden węzeł ukryty — pomieszczenie, którego szukamy — i po jednym węźle na każdy obiekt ze słownika. 

**[ANIMACJA]**
Wykryte obiekty (karty z YOLO) przekształcają się w węzły-koła na dole ekranu; dołączają do nich pozostałe obiekty ze słownika. U góry pojawia się **jeden** węzeł `Pomieszczenie` (zaokrąglony prostokąt), a w nim mały wykres słupkowy — rozkład prawdopodobieństwa po pomieszczeniach, na razie płaski (równomierny). Z węzła `Pomieszczenie` wyrastają **skierowane strzałki** w dół, po jednej do każdego obiektu — gwiazda. To jest prawdziwa struktura sieci.

---

**[NARRACJA]**
Założeniem modelu jest niezależność obiektórw od pomieszczenia w którym się znajdują. Z tego powodu nie ma krawędzi między obiektami.

**[ANIMACJA]**
Na chwilę pojawiają się czerwone łuki łączące pary obiektów (np. `klawiatura ↔ mysz`), po czym zostają przekreślone i znikają. Podpis: „Brak krawędzi między obiektami → naiwne założenie niezależności". Zostaje czysta gwiazda.

---

**[NARRACJA]**
Każdy obiekt jest zmienną binarną: obecny albo nieobecny. Inforamcje te przesuwają rozkład prawdopodobieństwa po pomieszczeniach. Po zebraniu wszystkich dowodów wygrywa jedno pomieszczenie: biuro.

**[ANIMACJA]**
Wykryte obiekty zapalają się na zielono z ptaszkiem (obecne), pozostałe gasną z czerwonym krzyżykiem (nieobecne). Z każdego węzła-obiektu w górę po strzałce płynie kropka do węzła `Pomieszczenie`; słupki rozkładu przestają być płaskie i `Biuro` wyrasta najwyżej. Na środku ekranu pojawia się podpis „Najbardziej prawdopodobne: Biuro".

---

## SCENA 4 — Reguła Bayesa i rola dowodów

**[NARRACJA]**
Dzięki założeniu o niezależności, prawdopodobieństwo pomieszczenia przy danych obiektach zapisujemy jako prosty iloczyn: prior pomieszczenia razy rozkład warunkowy dla każdego obiektu. Ale czy brak obiektu też może być dowodem?

**[ANIMACJA]**
Na ekranie pojawia się wzór (LaTeX), na razie w wersji podstawowej:

```
P(pomieszczenie | obiekty) ∝ P(pomieszczenie) · Π P(obiekt | pomieszczenie)
```

Pod wzorem pytanie: „Czy brak obiektu może być dowodem przeciw?". Wzór znika (jeszcze się nie zmienia).

---

**[NARRACJA]**
Wróćmy do grafu. Brak obiektu to też informacja. Brak łóżka to dowód przeciwko sypialni, brak kuchenki — przeciwko kuchni. A obecność monitora przechyla szalę na biuro.

**[ANIMACJA]**
Sieć-gwiazda buduje się tak samo jak wcześniej. Następnie po kolei: węzeł `łóżko` dostaje czerwony krzyżyk (nieobecny), czerwona kropka płynie w górę i słupek `Sypialnia` maleje; potem `kuchenka` dostaje czerwony krzyżyk i słupek `Kuchnia` maleje; na końcu `monitor` dostaje zielony znacznik (obecny), zielona kropka płynie w górę i słupek `Biuro` rośnie, wygrywając. Pojedynczy podpis na dole ekranu zmienia się przy każdym kroku: „Brak łóżka → dowód PRZECIW sypialni", „Brak kuchenki → dowód PRZECIW kuchni", „Monitor obecny → dowód ZA biurem".

---

**[NARRACJA]**
Wracamy do wzoru i go uzupełniamy: obok iloczynu po obiektach obecnych dochodzi iloczyn po obiektach nieobecnych — czynnik jeden minus prawdopodobieństwo. Zarówno obecność, jak i nieobecność jest dowodem.

**[ANIMACJA]**
Ten sam wzór podstawowy pojawia się ponownie, po czym płynnie przekształca się w wersję pełną:

```
P(pomieszczenie | obiekty) ∝ P(pomieszczenie)
        · Π_obecne P(obiekt | pomieszczenie)
        · Π_nieobecne (1 − P(obiekt | pomieszczenie))
```

Pod wzorem podpis: „Także brak obiektu jest dowodem."

---

## SCENA 5 — Trening: jak uczymy model

**[NARRACJA]**
Trening jest prosty. Dla każdego rodzaju pomieszczenia zliczamy, w ilu zdjęciach pojawił się dany obiekt. Dodatkowo liczymy ile zdjęć kategorii mamy.

**[ANIMACJA]**
Tabela **zliczeń** (liczby całkowite) — wiersze to pomieszczenia, kolumny to obiekty, a dodatkowa kolumna `# obrazów` po prawej podaje liczbę zdjęć danego pomieszczenia:

```
            krzesło  łóżko  piekarnik  zlew  ...  klawiatura  # obrazów
biuro          80      0        0       0   ...      30         208
sypialnia      32    138        0       0   ...       0         208
kuchnia        41      0       64      19   ...       0         208
korytarz        5      0        0       0   ...       0         208
```

Wiersze pojawiają się jeden po drugim. Podpis: „Zliczamy wystąpienia obiektów w pomieszczeniach".

---

**[NARRACJA]**
Wystarczyłoby teraz podzielić zliczenia przez liczbę zdjęć — i mamy prawdopodobieństwo. Jest jednak z tym problem. Jeśli jakiegoś obiektu nigdy nie widzieliśmy w danym pomieszczeniu, dostajemy zero. A wtedy jedna nietypowa detekcja na zawsze wyklucza tę kategorię. Dlatego stosujemy wygładzanie Laplace'a: do licznika i mianownika dodajemy małą liczbę alfa. Dzięki temu żaden obiekt nie ma prawdopodobieństwa dokładnie zero.

**[ANIMACJA]**
Podświetla się prawdziwe zero — komórka `biuro × łóżko = 0` (czerwona ramka). Pojawia się tytuł „Wygładzanie Laplace'a" i naiwny wzór, który daje zero:

```
P = n_obj / n_pokój = 0 / 208 = 0      ✗ obiekt staje się „niemożliwy" na zawsze
```

Wzór **płynnie przekształca się (morph)** w wersję wygładzoną — czynniki `+α` i `+2α` wrastają na swoje miejsca, a ramka komórki zmienia kolor z czerwonego na zielony:

```
P = (n_obj + α) / (n_pokój + 2α) = (0 + 1) / (208 + 2) ≈ 0.005
```

Podpis: „α = 1 → żaden obiekt nie ma P = 0".

---

**[NARRACJA]**
Teraz wystarczy zastosować ten wzór do każdej komórki. Zliczenia zamieniają się w prawdopodobieństwa warunkowe — i od razu widać, które obiekty są charakterystyczne dla których pomieszczeń.

**[ANIMACJA]**
Wzór przesuwa się (morph) w dół jako ogólna reguła. Liczby w całej tabeli morfują ze zliczeń w prawdopodobieństwa (np. `biuro × krzesło: 80 → 0.39`), a komórki zyskują kolor zależny od wartości:

```
            krzesło  łóżko  piekarnik  zlew  ...  klawiatura
biuro         0.39   0.00     0.00    0.00  ...     0.15
sypialnia     0.15   0.66     0.00    0.00  ...     0.00
kuchnia       0.20   0.00     0.31    0.10  ...     0.00
korytarz      0.03   0.00     0.00    0.00  ...     0.00
```

Na koniec dla każdego obiektu podświetla się (biała ramka) pomieszczenie o najwyższym prawdopodobieństwie.

---

## SCENA 6 — Wyniki treningu

**[NARRACJA]**
Zobaczmy wyniki treningu. Macierz pomyłek pokazuje, które pomieszczenia są ze sobą mylone. Na przekątnej widać poprawne klasyfikacje — im jaśniejsza komórka, tym wyższy recall. Pomieszczenia z charakterystycznym wyposażeniem, takie jak sypialnia czy kuchnia, są klasyfikowane dobrze. Najgorzej wypada korytarz — jego recall wynosi 2%, jest nieodróżnialny od schodów.

**[ANIMACJA]**
Pojawia się tytuł „Macierz pomyłek". Następnie wlatuje heatmapa 10×10 z wartościami znormalizowanymi wierszami (recall per room). Każda komórka z niezerową liczbą ma wartość liczbową — białą na ciemnym tle, szarą na jasnym. Pojawiają się etykiety wierszy (prawdziwa kategoria) i kolumn (przewidywana kategoria) wraz z podpisami osi; jednocześnie tytuł znika.

Podświetlają się komórki na przekątnej białą ramką. Na dole pojawia się napis: „Przekątna = poprawne klasyfikacje".

Następnie podświetlają się czerwoną ramką komórki poza przekątną w wierszu najgorzej sklasyfikowanego pomieszczenia. Napis morphuje w: „[nazwa]: recall = X% (pomylona z innymi!)".

Całość znika.

---

**[NARRACJA]**
Największy problem to pomieszczenia, które są do siebie podobne — mają te same obiekty, albo są po prostu puste. Pusty korytarz i pusta klatka schodowa wyglądają tak samo.

**[ANIMACJA]**
Dwa zdjęcia obok siebie — pusty korytarz i pusta klatka schodowa. Pod każdym etykieta z pytajnikiem. Na dole napis: „Brak charakterystycznych obiektów". Całość znika.

---

**[NARRACJA]**
Spójrzmy na metryki dla każdego pomieszczenia z osobna. Precyzja, czułość i F1 różnią się znacznie między kategoriami. Pomieszczenia takie jak sypialnia i łazienka osiągają wysokie wyniki, podczas gdy siłownia czy biblioteka — znacznie gorsze.

**[ANIMACJA]**
Pojawia się wykres słupkowy pogrupowany (precyzja / czułość / F1) dla każdego pomieszczenia. Słupki wyrastają od dołu z opóźnieniem. Legenda z kolorami pojawia się pod tytułem. Całość znika.

---

## SCENA 7 — Demo na własnych zdjęciach

**[NARRACJA]**
Sprawdźmy sami. Mamy zdjęcie salonu. Model jest bardzo pewny: widzi stół, krzesła kubek, miskę. To obiekty silnie charakterystyczne dla jadalni — sieć bayesowska nie ma wątpliwości. Może to ja źle nazywam ten pokój...

**[ANIMACJA]**
Zdjęcie salonu i output programu

---

**[NARRACJA]**
Następnie sypialnia. Tu też model radzi sobie dobrze: łóżko jest bardzo silnym sygnałem, na tyle że klawiatura czy telewizor nie przewarza na korzyść biura.

**[ANIMACJA]**
Zdjęcie pokoju i output programu

---

**[NARRACJA]**
Ostatni test - schody. I tu się pojawia problem. Na zdjęciu model nie widzi prawie nic poza kwiatami. Błędnie szacuje pomieszczenie ale jego pewność też nie jest za duża.

**[ANIMACJA]**
Zdjęcie schodów i output programu
```

---

## SCENA 8 — Porównanie z artykułem

**[NARRACJA]**
Nasz projekt jest próbą remplementacji artykułu BORM — Bayesian Object Relation Model for Indoor Scene Recognition — autorstwa Zhou i innych, opublikowanego w 2021.

**[ANIMACJA]**
Na ekranie pojawia się screenshot strony tytułowej artykułu (kadr z `artricle.pdf`): widoczny tytuł, autorzy, logo arXiv. Przez chwilę pozostaje jako tło.

---

**[NARRACJA]**
Autorzy poszli trochę inną drogą niż my. Zamiast wykrywać obiekty przez YOLO i operować na ich nazwach, używają modelu przetwarzania sceny wytrenowanego na zbiorze ADE20K. Taki model zwraca nie listę obiektów, ale pełny rozkład prawdopodobieństwa po 150 kategoriach — łóżko, ściana, podłoga, okno i tak dalej. To dużo bogatszy opis sceny kodujący także relacje między obiektami. Różnice te prowadzą do głębszej reprezentacji sceny jednak wymaga to więszkej ilości danych.

**[ANIMACJA]**
Dwa schematy obok siebie:

```
NASZ MODEL:                      ARTYKUŁ (BORM):
[Zdjęcie]                        [Zdjęcie]
   ↓                                 ↓
[YOLOv8m → lista obiektów]      [IOM (ADE20K) → wektor 150 cech]
   ↓                                 ↓
[Naive Bayes]                    [Bayesian Object Relation Model]
   ↓                                 ↓
[Pomieszczenie]                  [Pomieszczenie]
```

---


## SCENA 9 — Podsumowanie

**[NARRACJA]**
Jak nasz model wypada na tle artykułu? Porównajmy dokładność na tym samym zbiorze — Places365. Nasz model dorównuje podstawowemu modelowi z artykułu, opartemu wyłącznie na obiektach. Pełny model BORM, który dodaje relacje między obiektami, osiąga wyższą dokładność — to pokazuje, ile daje bogatsza reprezentacja sceny.

**[ANIMACJA]**
Tytuł „Porównanie metryk dokładności", podtytuł „Walidacja na zbiorze Places365". Trzy słupki wyrastają od dołu: `Artykuł (OM)` ≈ 47% (fioletowy), `Nasz model` (zielony, wyróżniony żółtą obwódką) z naszą dokładnością, `Artykuł (BORM)` ≈ 75% (fioletowy). Nad słupkami pojawiają się wartości procentowe. Na dole podpis: „Nasz model dorównuje modelowi podstawowemu z artykułu."

---

**[NARRACJA]**
Spójrzmy na wyniki metryki F1 dla każdego pomieszczenia. Naiwny Bayes oparty na wykrytych obiektach działa zaskakująco dobrze dla pomieszczeń z charakterystycznym wyposażeniem. Wyniki spadają, gdy pokoje są podobne wizualnie lub słabo wyposażone.

**[ANIMACJA]**
Tytuł „Wyniki — metryka F1". Wykres słupkowy F1 dla każdego pomieszczenia, posortowany malejąco; kolor słupka przechodzi płynnie od czerwonego (niskie F1) do zielonego (wysokie). Pod słupkami obrócone etykiety pomieszczeń, w tle pozioma siatka z wartościami 25 / 50 / 75 / 100%. Słupki wyrastają od dołu z opóźnieniem, potem pojawiają się wartości. Na dole podpis: „Ogólna dokładność: … Top-3: …".

---

**[NARRACJA]**
Kluczowa obserwacja: YOLO widzi tylko to, co jest w jego słowniku. Jeśli pokój jest charakterystyczny przez tekstury, kolory czy układ przestrzenny — te cechy są dla nas niewidoczne. To ograniczenie reprezentacji, nie modelu bayesowskiego.

**[ANIMACJA]**
Tytuł (pomarańczowy) „Ograniczenie — słownik obiektów YOLO". Po lewej zdjęcie biblioteki, pod nim czerwony napis `YOLO: (brak detekcji)`. Po prawej lista cech niewidocznych dla YOLO: tapety / tekstury ścian, rodzaj podłogi, układ przestrzenny, oświetlenie. Na dole podpis: „Ograniczeniem jest reprezentacja, nie model."

---

## SCENA 10 — Zakończenie / Credits

**[NARRACJA]**
Projekt inspirowany jest artykułem BORM — Bayesian Object Relation Model for Indoor Scene Recognition — autorstwa Zhou i innych, opublikowanym na konferencji IROS w 2021 roku. Dane treningowe pochodzą ze zbioru Places365. Do detekcji obiektów wykorzystaliśmy pre-trenowany model YOLOv8m. Link do repozytorium znajdziesz w opisie.

**[ANIMACJA]**
Napis z bibliografią pojawia się stopniowo:

```
Artykuł:
  L. Zhou et al., "BORM: Bayesian Object Relation Model
  for Indoor Scene Recognition," IROS 2021. arXiv:2108.00397

Dataset:
  Places365 (MIT CSAIL)

Detekcja obiektów:
  YOLOv8m (Ultralytics)

Repozytorium:
  github.com/newlochte/bayesian-room-recognition

Animacje:
  logo manim ce
```

---

**[NARRACJA]**
Dziękuję za obejrzenie.

**[ANIMACJA]**
Bibliografia znika. Wcentrum Wykonanie: Karolina Michalak i Tymoteusz Tomczak. Pojawia się mały tekst: „Projekt wykonany na zajęcia ze Sztucznej Inteligencji w Robotyce" oraz „Podziękowania dla pani Joanny Piasek-Skupnej". Następnie fade out — proste ciemne tło z dużym napisem: *Dziękuję*.

---

## Notatki realizacyjne

- Sceny 1–2: intro, może być szybkie (~30s każda)
- Sceny 3–5: techniczne serce filmu, tempo spokojniejsze (~60–90s każda)
- Scena 6–7: demo, dynamiczne (~45s każda)
- Scena 8: porównanie, krótkie (~30s)
- Sceny 9–10: outro (~45s łącznie)
- Łączny czas docelowy: ~8–12 minut
