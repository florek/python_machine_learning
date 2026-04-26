# pml

Repo pod naukę — ML, ćwiczenia z kodem, porządek w projekcie. Spokojnie, krok po kroku.

## Dokumentacja

[W notatkach](docs/lessons/wiedza.md) jest aktualna ściąga z kursu do zakresu strony 89: perceptron, Adaline (batch i SGD), standaryzacja, granice decyzji, sigmoida i podstawy regresji logistycznej.

Quiz (aktualna seria): [pytania](docs/quiz/questions/26.04.2026_questions.md), [moje odpowiedzi (szablon)](docs/quiz/my_answers/26.04.2026_my_answers.md), [klucz](docs/quiz/answers/26.04.2026_answers.md).

Wcześniejsze serie: [21.04.2026](docs/quiz/questions/21.04.2026_questions.md) ([szablon](docs/quiz/my_answers/21.04.2026_my_answers.md), [klucz](docs/quiz/answers/21.04.2026_answers.md)); [20.04.2026](docs/quiz/questions/20.04.2026_questions.md) ([szablon](docs/quiz/my_answers/20.04.2026_my_answers.md), [klucz](docs/quiz/answers/20.04.2026_answers.md)); [19.04.2026](docs/quiz/questions/19.04.2026_questions.md); [15.04.2026](docs/quiz/questions/15.04.2026_questions.md).

## Start w kilku krokach

```text
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

W Cursorze / VS Code wybierz interpreter z `venv` (albo zostaw domyślny z `.vscode/settings.json`), żeby Pylance/basedpyright widział zainstalowane paczki (`pandas`, `matplotlib`).

Lekcje z kodem obejmują sekwencję perceptron -> Adaline -> sigmoid/log-loss -> regresja logistyczna oraz narzędzia pomocnicze do standaryzacji i rysowania granic decyzji.
