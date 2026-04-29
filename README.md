# pml

Repo pod naukę — ML, ćwiczenia z kodem, porządek w projekcie. Spokojnie, krok po kroku.

## Dokumentacja

[W notatkach](docs/lessons/wiedza.md) jest aktualna ściąga z kursu do zakresu strony 97: m.in. perceptron, Adaline (batch i SGD), standaryzacja, granice decyzji, sigmoida, log-loss (także w odniesieniu do osi prawdopodobieństwa), regresję logistyczną z batch GD i ucięciem argumentu sigmoidy, dane przez API scikit-learn, wybór dwóch cech po indeksach oraz etykiety 0/1 w zadaniu binarnym z wieloklasowego źródła.

Quiz (aktualna seria): [pytania](docs/quiz/questions/29.04.2026_questions.md), [moje odpowiedzi (szablon)](docs/quiz/my_answers/29.04.2026_my_answers.md), [klucz](docs/quiz/answers/29.04.2026_answers.md).

Wcześniejsze serie: [27.04.2026](docs/quiz/questions/27.04.2026_questions.md); [21.04.2026](docs/quiz/questions/21.04.2026_questions.md) ([szablon](docs/quiz/my_answers/21.04.2026_my_answers.md), [klucz](docs/quiz/answers/21.04.2026_answers.md)); [20.04.2026](docs/quiz/questions/20.04.2026_questions.md) ([szablon](docs/quiz/my_answers/20.04.2026_my_answers.md), [klucz](docs/quiz/answers/20.04.2026_answers.md)); [19.04.2026](docs/quiz/questions/19.04.2026_questions.md); [15.04.2026](docs/quiz/questions/15.04.2026_questions.md).

## Start w kilku krokach

```text
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

W Cursorze / VS Code wybierz interpreter z `venv` (albo zostaw domyślny z `.vscode/settings.json`), żeby Pylance/basedpyright widział zainstalowane paczki (`pandas`, `matplotlib`).

Lekcje z kodem obejmują sekwencję perceptron -> Adaline -> sigmoid/log-loss -> regresja logistyczna oraz narzędzia pomocnicze do standaryzacji i rysowania granic decyzji.
