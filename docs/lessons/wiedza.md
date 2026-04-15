# Notatki z kursu — od czego zaczynamy

Luźny styl, ale treść ma być ściągą: definicje, „dlaczego tak”, typowe pułapki. Sama wiedza — bez mapowania na konkretne pliki projektu.

## Po co ćwiczymy ML w Pythonie

Uczymy się łączyć matematykę klasyfikacji z kodem: dane jako wektory cech, model jako prosta funkcja decyzyjna, trening jako powtarzalna aktualizacja parametrów. Pierwszy krok to często **perceptron** — najprostszy klasyfikator liniowy, który dobrze tłumaczy ideę **granicy decyzji** i **uczenia online** (aktualizacja po jednej próbce). W praktyce ten sam schemat „wejście liniowe → decyzja” powtarza się w większych modelach: pojedynczy perceptron to jak jeden neuron liniowy z progiem; sieci składają się z wielu takich bloków (zwykle z **nieliniowością** między warstwami), a współczesne modele sekwencyjne też na tym się opierają — tylko w skali i z inną architekturą.

## Środowisko i narzędzia

**Wirtualne środowisko** izoluje wersje bibliotek od systemu i od innych projektów. Aktywujesz je lokalnie, instalujesz paczki tylko „do środka” tego katalogu — mniej konfliktów, powtarzalne uruchomienia u Ciebie i u innych, jeśli współdzielicie listę zależności.

**Plik zależności** to zamrożona lista nazw pakietów i często minimalnych wersji. Zamiast pamiętać „zainstaluj to i tamto”, robisz jedną komendę instalacji z listy — standard w małych i większych projektach.

**Ignorowanie katalogu środowiska w kontroli wersji** ma sens, bo w środku są tysiące plików binarnych i środowiskowych; repozytorium trzyma **przepis** (lista paczek), a nie **gotową kopię** zainstalowanego świata. Klonujesz projekt, odtwarzasz wirtualne środowisko u siebie — lekko i czytelnie.

**Analiza statyczna w edytorze** (podkreślenia importów) patrzy na **ten interpreter Pythona**, który jest wybrany w IDE. Jeśli paczki są zainstalowane tylko w wirtualnym środowisku, a edytor wskazuje na inny Python, zobaczysz fałszywe alarmy — rozwiązanie to ten sam interpreter, w którym faktycznie instalujesz zależności.

## Dane tabelaryczne i wczytywanie

**Zbiór iris** ma kilkadziesiąt próbek na klasę, cztery cechy numeryczne i etykietę gatunku. Na początku często bierze się **dwie klasy** i **dwie cechy**, żeby wizualizacja była w 2D.

**Wczytanie z dysku zamiast z adresu URL** ma sens przy ćwiczeniu: odczyt przez bibliotekę z sieci wymaga działającego DNS i połączenia; błąd typu „nie udało się rozwiązać nazwy hosta” oznacza problem sieciowy, nie matematyczny modelu.

**Mapowanie etykiet tekstowych na liczby** dla dwóch klas: jedna klasa jako wartość ujemna, druga jako dodatnia (np. `-1` i `1`) — zgodnie z regułą uczenia perceptronu.

**Wybór podzbioru wierszy i kolumn** w ramce danych: pierwsze N wierszy na trening binarny, konkretne kolumny jako współrzędne punktu w przestrzeni cech (np. długość działki kielicha i długość płatka).

## NumPy w roli „silnika” pod model

**Tablice** reprezentują wektory cech i wagi; operacje typu iloczyn skalarny robi się funkcjami zoptymalizowanymi pod CPU, zamiast ręcznych pętli po elementach w czystym Pythonie.

**Generator z ustalonym ziarnem** daje powtarzalny start losowych wag: ten sam seed → te same liczby przy kolejnym uruchomieniu. Przydaje się przy debugowaniu i porównywaniu eksperymentów.

**Małe wartości startowe wag** z rozkładu normalnego (blisko zera, niska wariancja) to typowy trik: unikasz zbyt dużych aktywacji na wejściu i „płaskiego” startu treningu.

**Kombinacja liniowa** cech i wag plus bias to dokładnie to, co liczy się przed progiem: suma ważonych wejść i przesunięcie. To serce warstwy liniowej w wielu modelach.

**Prog warunkowy na całych tablicach** (`wartość jeśli warunek, w przeciwnym razie inna`) pozwala w jednym kroku przypisać klasę tam, gdzie wartość przed progiem jest nienegatywna, i drugą klasę w przeciwnym razie — wygodne przy wektorze predykcji i przy **kodowaniu etykiet** z nazw klas.

**Siatka punktów w 2D** do rysowania granicy: dwie osie zakresów, **iloczyn kartezjański** zakresów przez funkcję generującą siatkę, spłaszczenie do listy punktów, predykcja klasyfikatora dla każdego punktu, **przekształcenie z powrotem** do kształtu siatki — pod **wypełnione kontury** kolorami klas.

## Matplotlib — wizualizacja w ćwiczeniu

**Wykres rozrzutu** dwóch cech: osobne serie dla klas (kolor, marker), oś X i Y z opisem fizycznym cechy.

**Wykres błędu w czasie treningu** oś pozioma to numer epoki (od jedynki), pionowa to liczba **niezerowych korekt** w epoce — sygnał, czy model jeszcze się poprawia.

**Regiony decyzji:** tło jako wypełnione kontury z lekką przezroczystością, na wierzchu te same próbki co wcześniej; **mapa kolorów** ograniczona do tylu kolorów, ile jest **unikalnych etykiet** po treningu.

**Backend bez interakcji** (np. rasteryzacja do pliku): wywołanie pokazujące okno może dać ostrzeżenie lub nic nie pokazać — to kwestia środowiska uruchomienia, nie samego modelu.

## Perceptron — mechanika

**Etykiety** w tym wariancie to dwie klasy wyrażone liczbowo: dodatnia i ujemna (np. `1` i `-1`). Taki wybór upraszcza regułę aktualizacji: znak błędu od razu mówi, w którą stronę pchnąć wagę.

**Bias** można trzymać jako osobny parametr albo **dokleić sztuczną cechę** i traktować go jak kolejną wagę — w kodzie często widać wektor wag dłuższy o jeden element: pierwszy to próg, reszta to wagi cech.

**Reguła uczenia:** dla każdej próbki liczysz różnicę między prawdziwą etykietą a aktualną predykcją, mnożysz przez **współczynnik uczenia** (często oznaczany grecką etą) i dodajesz do wag składową proporcjonalną do wejścia; bias aktualizujesz tak, jakby wejście „stałej jedynki” było zawsze obecne — czyli sam skalar tej aktualizacji ląduje na progu.

**Pętla zewnętrzna — epoki:** jedna epoka to przejście po wszystkich próbkach (w ustalonej kolejności w najprostszym wariancie). Parametr liczby epok ogranicza czas treningu; za mało — model nie zdąży się nauczyć, za dużo — przy danych nieseparowalnych liniowo i tak nie „zamazuje” błędu do zera, a koszt obliczeń rośnie.

**Predykcja przed progiem:** jeśli **suma ważona + bias** jest większa lub równa zero, przypisujesz jedną klasę; w przeciwnym razie drugą. Granica to hiperpłaszczyzna w przestrzeni cech.

**Licznik „popsutych” aktualizacji w epoku:** zliczasz, ile razy w danej epoce krok uczenia był niezerowy (czyli model jeszcze poprawiał się na próbce). To nie jest dokładność na zbiorze testowym, ale **sygnał, czy trening jeszcze coś zmienia** — przydatny przy szybkim podglądzie zbieżności.

## Pułapki i dobre nawyki

**Separowalność liniowa:** klasyczny perceptron ma gwarancje sensowne tylko wtedy, gdy klasy da się oddzielić płaszczyzną. Przy nakładających się chmurach punktów błąd treningowy może oscylować — wtedy inne modele lub cechy są konieczne.

**Współczynnik uczenia:** za duży — oscylacje i niestabilność; za mały — powolny postęp. W praktyce zwykle eksperymentalnie lub z rozszerzeniami (np. później: regularyzacja, inne reguły optymalizacji).

**Powtarzalność:** ustalone ziarno losowe i ta sama kolejność epok dają powtarzalny przebieg na tym samym kodzie i danych — ułatwia porównanie dwóch wersji algorytmu.

**Spójność środowiska:** jeden Python do instalacji paczek i do uruchamiania skryptów; ten sam interpreter w IDE co w terminalu — mniej chaosu z importami i wersjami.

**Dane lokalne vs sieć:** ćwicząc wczytywanie tabeli, trzymaj kopię pliku u siebie, jeśli chcesz uniknąć zależności od połączenia i DNS w momencie nauki.
