# Функція Розпізнавання Кон'юнкції за допомогою Нейронної Мережі

Цей проект реалізує нейронну мережу для розпізнавання функції кон'юнкції. Нейронна мережа тренується для передбачення виходу на основі заданих вхідних даних.

## Опис

Архітектура нейронної мережі складається з трьох шарів: двох прихованих шарів і одного вихідного шару. Кожен прихований шар містить два нейрони. Функція активації, яка використовується, - це сигмоїдна функція.

Процес навчання включає ітерацію через набір навчальних даних і коригування ваг і зміщень нейронів за допомогою зворотного поширення для мінімізації помилки між передбаченим виходом і фактичним цільовим виходом.

## Використання

Щоб використовувати нейронну мережу:

1. Клонуйте цей репозиторій на ваш локальний комп'ютер.
4. Натисніть кнопку "Компілювати", щоб побачити передбачений результат.

## Результати

Нейронна мережа тренується з використанням 20,000 ітерацій. Однак важливо зазначити, що прогнози можуть бути не завжди точними через складність функції кон'юнкції та обмеження архітектури нейронної мережі.
