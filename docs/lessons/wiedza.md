# Notatki z kursu ML (zakres do strony 94)

## Co obejmuje ten zakres

Zakres obejmuje fundamenty uczenia maszynowego: rodzaje uczenia, przygotowanie danych, perceptron, Adaline w wariancie batch i SGD, standaryzację cech, wizualizację granic decyzji oraz regresję logistyczną: sigmoida, log-loss, gradient prosty w wersji skryptowej, porównanie kosztów w zależności od etykiety, oraz wykorzystanie interfejsu scikit-learn do wczytania przykładowych danych i konstruowania zadania binarnego wśród wielu klas.

## Rodzaje uczenia maszynowego

Uczenie nadzorowane bazuje na parach wejście-wyjście i obejmuje klasyfikację oraz regresję. Uczenie nienadzorowane działa bez etykiet i służy m.in. do klasteryzacji oraz redukcji wymiarowości. Uczenie przez wzmacnianie opiera się na sygnale nagrody i sekwencji decyzji podejmowanych przez agenta.

## Kluczowe pojęcia

Cechy to kolumny wejściowe, a zmienna celu to przewidywany wynik. Funkcja straty opisuje błąd pojedynczej próbki, a funkcja kosztu agreguje błąd dla zbioru. Hiperparametry, takie jak współczynnik uczenia i liczba epok, nie są uczone bezpośrednio z danych i silnie wpływają na dynamikę treningu.

## Strategia pracy z danymi

Standardowy przepływ to: przygotowanie danych, podział na zbiory, trening, walidacja i test końcowy. Parametry skalowania oblicza się na zbiorze treningowym i dopiero potem stosuje do pozostałych danych. To ogranicza ryzyko wycieku informacji i zawyżonych wyników.

## Środowisko i narzędzia

Wirtualne środowisko izoluje zależności i minimalizuje konflikty wersji. Lista zależności pozwala odtworzyć identyczny setup w innym miejscu. Jeśli IDE wskazuje inny interpreter niż terminal, pojawiają się fałszywe błędy importu mimo poprawnie zainstalowanych pakietów.

## Dane Iris i reprezentacje etykiet

Zbiór Iris ma 150 rekordów, cztery cechy numeryczne i etykietę gatunku. Cechy można wybierać po indeksach, np. dwie wybrane współrzędne do wizualizacji 2D albo długości działki kielicha i płatka. W wersji binarnej z pliku CSV etykiety tekstowe zwykle mapuje się do wartości -1 i 1 dla perceptronu i Adaline. W wersji wieloklasowej z gotowego API etykiety bywają kodowane 0, 1, 2; wówczas progowanie decyzji w regresji logistycznej (np. próg 0,5 na prawdopodobieństwie) dotyczy interpretacji 0/1, a wycinek tylko dwóch klas uzyskuje się przez odfiltrowanie wierszy należących do tych etykiet, bez zmiany samych etykiet, jeśli w zadaniu mają pozostać 0 i 1.

Wspólne dla obu reprezentacji: spójność między zakresem etykiet a postacią funkcji straty. Perceptron/Adaline w ćwiczeniu używają często -1 i 1, a logarytmiczna funkcja kosztu przy regresji logistycznej wygodnie współgra z 0 i 1, bo występują w niej współczynniki (1−y) oraz y mnożące składniki logarytmów.

## NumPy jako podstawa obliczeń

Model liniowy opiera się na iloczynie skalarnym cech i wag oraz składniku bias. Wektoryzacja eliminuje ręczne pętle i przyspiesza obliczenia. Ustalony seed generatora losowego zapewnia powtarzalność inicjalizacji wag i porównywalność eksperymentów.

## Perceptron: mechanika i ograniczenia

Perceptron używa progowej funkcji decyzji i klasyfikuje do dwóch klas. Aktualizacja wag odbywa się po pojedynczej próbce na podstawie różnicy między etykietą a predykcją. Liczba niezerowych aktualizacji w epoce działa jako szybki sygnał postępu uczenia.

Perceptron zbiega dla danych liniowo separowalnych. Gdy klasy nie są separowalne liniowo, liczba błędów może oscylować i wtedy limit epok jest konieczny.

## Adaline: przejście do optymalizacji gradientowej

Adaline używa liniowej aktywacji podczas uczenia i minimalizuje ciągłą funkcję kosztu SSE. W wersji batch aktualizacja wag odbywa się raz na epokę z gradientu policzonego na całym zbiorze. W wersji SGD aktualizacja następuje po każdej próbce, a historia uczenia zwykle zapisuje średni koszt epoki.

Tasowanie próbek przed epoką w SGD zmniejsza wpływ kolejności danych i poprawia stabilność uczenia. Metoda `partial_fit` umożliwia uczenie przyrostowe bez restartu wag.

## Standaryzacja cech

Standaryzacja do średniej 0 i odchylenia 1 pomaga, gdy cechy mają różne skale. Dla metod opartych na gradiencie zwykle daje szybszą i stabilniejszą zbieżność oraz zmniejsza ryzyko rozbiegania kosztu przy większym kroku uczenia.

## Sigmoida i pomost do regresji logistycznej

Sigmoida mapuje dowolną wartość rzeczywistą do przedziału (0, 1), więc nadaje się do modelowania prawdopodobieństwa klasy. W okolicy zera jest najbardziej czuła, a na krańcach nasyca się blisko 0 lub 1. Wykreślając koszt względem prawdopodobieństwa φ(z) dla etykiet 0 i 1, widać, że gdy właściwa klasa ma 1, kara rośnie, gdy model przypisuje niskie p; gdy właściwa to 0, kara rośnie przy wysokim p.

W regresji logistycznej sumaryczny koszt dopasowuje do zbioru jako suma składników postaci −y log(φ) i −(1−y) log(1−φ) dla etykiet 0/1, co silnie penalizuje błędnie, „pewne” predykcje.

Ograniczanie argumentu wykładniczego w sigmoidzie, np. przez ucięcie do rozsądnego przedziału, zabezpiecza przed przepełnieniem w typowej postaci 1/(1+exp(−z)).

## Implementacja regresji logistycznej z pełnej próbki (batch GD)

W wersji wsadowej gradient liczy się z całej macierzy próbek: błąd to różnica etykiet 0/1 a aktywacji sigmoidalnej, aktualizacja wag przypomina strukturę Adaline, lecz z nieliniową aktywacją i kosztem opartym o logi prawdopodobieństw. Dyskretną etykietę klasy uzyskuje się przez progowanie prawdopodobieństwa, typowo z progiem 0,5.

Kod wizualizujący regiony decyzji oczekuje obiektu z metodą przewidującą etykiety klas na macierzy punktów siatki, niezależnie od wewnętrznej reprezentacji modelu, o ile publiczne API to zachowuje. To wzorzec „kaczej typizacji” w małym ćwiczeniu: ważne są kontrakty wejścia-wyjścia, nie ręczna weryfikacja typu w runtime.

## Wczytywanie danych: plik lokalny a API

Lokalny plik CSV wymaga jawnego czytania, nagłówków i ręcznego opisu etykiet. Funkcja zbiorcza zwracająca macierz cech i wektor etykiet skraca schemat, gdy zbiór jest wbudowany. Oba warianty można konsekwentnie łączyć z tymi samymi późniejszymi krokami, jeśli dopasuje się etykiety do wybranej funkcji kosztu i decyzji.

## Wizualizacja modeli

Wykres rozrzutu pozwala zobaczyć separowalność klas w przestrzeni cech. Wykres kosztu lub liczby błędów na epokę pokazuje przebieg uczenia. Regiony decyzji powstają przez predykcję na siatce punktów 2D i narysowanie mapy klas na tle danych. Opcjonalnie wyróżnia się testowy podzbiór punktami o innej obwódce, jeśli taki wariant jest w skrypcie.

## Najczęstsze pułapki

Za duży współczynnik uczenia prowadzi do oscylacji i niestabilności kosztu. Brak standaryzacji utrudnia zbieżność w metodach gradientowych, gdy cechy mają skale o rzędach wielkości. Mieszanie środowisk Pythona powoduje fałszywe błędy importu.

Błędy sieciowe przy pobieraniu danych z internetu nie oznaczają błędu modelu, tylko problem z dostępnością źródła. Dlatego lokalna kopia danych jest praktyczna podczas nauki.

Dwa różne schematy etykiet (-1/1 oraz 0/1) łatwo pomylić: koszt i próg decyzji muszą odpowiadać tej samej konwencji, co etykiety w danym ćwiczeniu. Oś wykresu należy opisać tym, co faktycznie przedstawia cecha (w tym, czy została ustandaryzowana, czy nie), żeby interpretacja wizualizacji była wiarygodna.

## Pytania kontrolne do utrwalenia

Dlaczego standaryzacja cech często poprawia zachowanie metod gradientowych? Co zmienia przejście z progowej decyzji perceptronu do probabilistycznej interpretacji wyjścia przez sigmoidę? Kiedy wybór kosztu kwadratowego ma sens, a kiedy logarytmiczny, jeśli celem jest klasyfikacja probabilistyczna? Jak odróżnić etykietowanie 0/1 od -1/1 w pętli uczącej?
