# Notatki z kursu ML (zakres do strony 89)

## Co obejmuje ten zakres

Zakres obejmuje fundamenty uczenia maszynowego: rodzaje uczenia, przygotowanie danych, perceptron, Adaline w wariancie batch i SGD, standaryzację cech, wizualizację granic decyzji oraz wejście w regresję logistyczną przez sigmoidę i funkcję kosztu log-loss.

## Rodzaje uczenia maszynowego

Uczenie nadzorowane bazuje na parach wejście-wyjście i obejmuje klasyfikację oraz regresję. Uczenie nienadzorowane działa bez etykiet i służy m.in. do klasteryzacji oraz redukcji wymiarowości. Uczenie przez wzmacnianie opiera się na sygnale nagrody i sekwencji decyzji podejmowanych przez agenta.

## Kluczowe pojęcia

Cechy to kolumny wejściowe, a zmienna celu to przewidywany wynik. Funkcja straty opisuje błąd pojedynczej próbki, a funkcja kosztu agreguje błąd dla zbioru. Hiperparametry, takie jak współczynnik uczenia i liczba epok, nie są uczone bezpośrednio z danych i silnie wpływają na dynamikę treningu.

## Strategia pracy z danymi

Standardowy przepływ to: przygotowanie danych, podział na zbiory, trening, walidacja i test końcowy. Parametry skalowania oblicza się na zbiorze treningowym i dopiero potem stosuje do pozostałych danych. To ogranicza ryzyko wycieku informacji i zawyżonych wyników.

## Środowisko i narzędzia

Wirtualne środowisko izoluje zależności i minimalizuje konflikty wersji. Lista zależności pozwala odtworzyć identyczny setup w innym miejscu. Jeśli IDE wskazuje inny interpreter niż terminal, pojawiają się fałszywe błędy importu mimo poprawnie zainstalowanych pakietów.

## Dane Iris i przygotowanie pod zadania binarne

Zbiór Iris ma 150 rekordów, cztery cechy numeryczne i etykietę gatunku. W klasycznym ćwiczeniu binarnym wykorzystuje się dwie klasy i dwie cechy, co umożliwia wizualizację 2D. Etykiety tekstowe mapuje się do wartości liczbowych, najczęściej -1 i 1 dla perceptronu i Adaline.

W wersji wieloklasowej wygodnie jest pobrać cechy i etykiety bezpośrednio jako macierze NumPy, gdzie klasy mają kody 0, 1, 2.

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

## Wizualizacja modeli

Wykres rozrzutu pozwala zobaczyć separowalność klas w przestrzeni cech. Wykres kosztu lub liczby błędów na epokę pokazuje przebieg uczenia. Regiony decyzji powstają przez predykcję na siatce punktów 2D i narysowanie mapy klas na tle danych.

## Sigmoida i pomost do regresji logistycznej

Sigmoida mapuje dowolną wartość rzeczywistą do przedziału (0, 1), więc nadaje się do modelowania prawdopodobieństwa klasy. W okolicy zera jest najbardziej czuła, a na krańcach nasyca się blisko 0 lub 1.

W regresji logistycznej koszt dla poprawnej klasy 1 ma postać `-log(p)`, a dla klasy 0 `-log(1-p)`. Taka funkcja silnie karze błędne, pewne predykcje i jest lepiej dopasowana do klasyfikacji probabilistycznej niż SSE.

## Implementacja regresji logistycznej z GD

W praktyce można trenować model iteracyjnie przez gradient po wszystkich próbkach i aktualizować wagi analogicznie do Adaline, ale z aktywacją sigmoidalną i kosztem log-loss. W obliczeniu sigmoidy warto ograniczać argument funkcji wykładniczej przez `clip`, żeby uniknąć przepełnień numerycznych.

Próg decyzyjny 0.5 zamienia prawdopodobieństwo na etykietę klasy binarnej 0 lub 1.

## Najczęstsze pułapki

Za duży współczynnik uczenia prowadzi do oscylacji i niestabilności kosztu. Brak standaryzacji utrudnia zbieżność w metodach gradientowych. Mieszanie środowisk Pythona powoduje fałszywe błędy importu.

Błędy sieciowe przy pobieraniu danych z internetu nie oznaczają błędu modelu, tylko problem z dostępnością źródła. Dlatego lokalna kopia danych jest praktyczna podczas nauki.

## Pytania kontrolne do utrwalenia

Dlaczego standaryzacja cech często poprawia zachowanie metod gradientowych? Co zmienia przejście z progowej decyzji perceptronu do probabilistycznej interpretacji wyjścia przez sigmoidę? Kiedy metryka liczby błędnych aktualizacji wystarcza, a kiedy trzeba przejść na analizę kosztu i walidacji?
