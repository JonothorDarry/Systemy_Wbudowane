# Systemy_Wbudowane
Zasady kompilacji:

python setup.py build_ext --inplace


python FindSymbols.py ../kacykowanie/zdjs/Trivial/DSC_0054.JPG # przykładowe miejsce - pojedynczy obraz

Problemy:
1) Na razie część cythonowa nie jest satysfakcjonująco szybka: 4-6 sekund
2) Obracanie obrazka - nie wiem, czy obrócić obrazek o 0 czy 180 stopni, to raczej zobaczy się na labach, czy kamerę można ustawiać odpowiednio.
3) Napisz, czy python setup.py build_ext --inplace Ci działa- mi z moją anacondą wyskakiwało, do wyboru do koloru dla innych setupów albo brak numpy/arrayobject.h, albo Python.h (nawet z wersją developerską 3.7), albo nieznajomość formatu .pyx. Ten mi działa, ale to nie jest normalna konfiguracja żadną miarą.

Format:
1) Linii jest tyle, ile staffów
2) linia zaczyna się od trebla - TR albo bass clefa - CL
3) Nuty 1-8 oznaczam liczbą i literą
4) krzyżyk - nie wiem, co w nim ma być, więc format to litera+'#'

BTW - jakbyś chciał wiedzieć, to w cythonie tak się robi passa 2-wymiarowej tablicy przez referencję do funkcji:
https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html#c-and-fortran-contiguous-memoryviews
Segment kodu, który wykonywał się w kodzie przez 5 sekund, miał w sumie niecałe 700.000 wywołań - ale każde wywołanie robiło passa 2-wymiarowej funkcji przez jakąś para-refernecję, która działała jak referencja, wyglądała jak referencja, ale zajmowała jakąś tysiąckrotność czasu wykonania zwykłej referencji - pewnie to był pass 1000 referencji na kolejne jednowymiarowe tablice, ale mogę się mylić.
