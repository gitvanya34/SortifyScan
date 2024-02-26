# SortifyScan: Детекция и сегментация товаров на сортировочной ленте

<a href=""><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
## Описание
Этот репозиторий посвящен разработке системы детекции и сегментации товаров на сортировочной ленте в контексте складской логистики. Проект направлен на создание инструмента для автоматической обработки видеопотока с камер, установленных на сортировочных лентах, с целью идентификации и сегментации различных товаров.

[Подробная информация об исследовании](https://github.com/gitvanya34/SortifyScan/blob/main/%D0%9F%D0%BE%D1%8F%D1%81%D0%BD%D0%B8%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D0%B0%D1%8F%20%D0%B7%D0%B0%D0%BF%D0%B8%D1%81%D0%BA%D0%B0.pdf)

## Функциональность
- Детекция товаров на видеопотоке с камер на сортировочной ленте.
- Сегментация обнаруженных товаров для последующей идентификации и классификации.

## Актуальность 
В складской логистики задача детекции играет критически важную роль, так как она лежит в основе автоматизации процессов учета и управления запасами. Автоматическое распознавание товаров и их характеристик через компьютерное зрение позволяет значительно ускорить и оптимизировать логистические процессы.

## Технологии 
- Python 3.x
- Ultralytics
- OpenCV
- YOLOv8
- SAM (Segment Anything Model)

## Графики обучений 
Моделью старта возьмем версию весов YOLOv8l.pt
В результате обучения стартовой модели 10-ю эпохами получаем обновлённый файл весов best.pt с метриками 

<p>
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/395831f6-e25c-4b24-a96e-7bf8583b8630" width="70%" alt="image">
</p>

  Далее приступаем к дообучению модели. В качестве датасета можно ис-пользовать как предыдущий датасет, так и другой подходящий с такой разметкой. Информация по сведению датасетов к одной аннотации рассказано в следующем пункте.
Для дообучения модели в функции обучения в качестве аргумента весов используем best.pt из предыдущего запуска.
В результате дообучения модели 15-ю эпохами получаем обновленный файл весов best.pt с метриками 

<p>
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/db4d9a7f-99bf-4613-a8b2-146b10f4e5e9" width="70%" alt="image">
</p>

## Детекция

<p>
  <img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/713bb53b-e7f8-4aff-aab8-4904a3fa13e3" width="40%" alt="image">
  <img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/70fac346-736f-4904-b78f-066dab733ac5" width="30%" alt="image">
</p> 
<p>
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/f7e270f4-126f-4497-92ae-535e6b482127" width="30%" alt="image">
</p> 

## Сегментация 

<p>
 <img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/ebd9fed5-f74b-4a8b-b220-6203d162f80d" width="50%" alt="image">
 <img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/07207efc-aadd-4b39-8fc3-3bb9a274258a" width="30%" alt="image">
</p> 


## Планы развития

В данный момент разрабатывается полноценный сервис API для взаимодействия между клиентным приложением и серверной частью 

<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/2b846e35-62cf-48b0-9686-a04f11dc0085" width="30%" alt="image">
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/ea969d09-b638-49e3-b85a-ad0817ca2919" width="60%" alt="иаграмма без названия drawio">

<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/5600ab6b-fbef-4f6d-b111-5592e3c80df3" width="50%" alt="edf9ce6c">
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/022e00d7-7943-4162-94af-46d8ef9a09e8" width="50%" alt="схема drawio">

## DemoGif


