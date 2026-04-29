# Historia wyników

| Data | Wynik | % | Braki | Link |
|------|-------|---|-------|------|
| 29.04.2026 | 6/20 | 30% | 5, 6, 16, 20 (X) | [wynik](29.04.2026_results.md) |
| 27.04.2026 | 15/20 | 75% | 4, 6, 8, 16, 20 (X) | [wynik](27.04.2026_results.md) |
| 26.04.2026 | 16/20 | 80% | 2, 10, 17, 19 (niedozwolona litera X) | [wynik](26.04.2026_results.md) |
| 24.04.2026 | 15/20 | 75% | 5, 12, 15, 17 (X); pyt. 18: błędna litera | [wynik](24.04.2026_results.md) |
| 21.04.2026 | 14/20 | 70% | 2, 7, 11, 12, 15, 17 (niedozwolona litera X) | [wynik](21.04.2026_results.md) |
| 20.04.2026 | 18/20 | 90% | 1, 18 (niedozwolona litera X) | [wynik](20.04.2026_results.md) |
| 19.04.2026 | 18/20 | 90% | 9, 19 (niedozwolona litera X) | [wynik](19.04.2026_results.md) |
| 15.04.2026 | 19/20 | 95% | brak (pyt. 7: niedozwolona litera X) | [wynik](15.04.2026_results.md) |

## Statystyki

- **Najlepszy wynik:** 19/20 (95%) — 15.04.2026
- **Najgorszy wynik:** 6/20 (30%) — 29.04.2026
- **Średnia ze wszystkich prób (8):** 15,125/20 (75,625%) — suma punktów 121, 121÷8=15,125
- **Ostatnie 5 wyników:** 6/20 (29.04.2026), 15/20 (27.04.2026), 16/20 (26.04.2026), 15/20 (24.04.2026), 14/20 (21.04.2026)

## Najczęstsze obszary do poprawy

Na podstawie wszystkich plików `*_results.md` (8 prób), zliczanie tagów w szczegółach błędów:

| Kategoria (tag) | Zgrubna liczba wystąpień błędów |
|-----------------|---------------------------------|
| [ML] | dominuje (w tym seria 29.04.2026: wiele pomyłek na paradygmatach uczenia, pipeline skalowania, perceptron/Adaline/logistyczna, kontrakt `predict`) |
| [SciKit-learn] | kilka (m.in. wybór kolumn vs klasy w Iris) |
| [NumPy] | kilka (m.in. stabilizacja sigmoidy) |
| [Matplotlib] | kilka (interpretacja wykresu kosztu vs φ(z)) |

Najmocniejsze wzorce: wpisy **X** zamiast A–D (liczone jak brak), mylenie **wyboru kolumn** z **filtrowaniem klas**, **RL** z nadzorem, **skalowanie test vs train**, **próg na aktywacji** vs inne reguły decyzji, rola **`partial_fit`**. Seria **29.04.2026** obniżyła średnią — warto powtórzyć sekcje notatek o Iris (API vs CSV), kosztach i procedurze preprocessingu.

## Notatki

Ranking kategorii nabiera sensu, gdy pojawią się kolejne pliki `*_results.md` i zróżnicowane tagi błędów.

We wcześniejszych wersjach tabeli „Historia” wszystkie niezaliczone pytania z quizu 21.04.2026 to były wyłącznie „X” przy odpowiedziach z bloku str. 70–79 (pyt. 11, 12, 15, 17) oraz definicji straty/kosztu, CV i `return_X_y` (pyt. 2, 7, 15).
