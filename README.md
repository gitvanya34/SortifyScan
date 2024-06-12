# SortifyScan: Детекция и сегментация товаров на сортировочной ленте

<a href="https://github.com/gitvanya34/SortifyScan/blob/main/demosortifyscan.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Py"></a>
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
YOLO (You Only Look Once) - это алгоритм для обнаружения объектов на изображениях с использованием глубокого обучения и нейронных сетей.
* Модель YOLO обучалась на размеченных датасетах в течение 10+15 эпох.
* Обучение проводилось на разнообразных изображениях товаров.
* Процесс обучения включал 10 эпох на одном датасете и 15 эпох на другом для улучшения точности детекции.
* Модель обучалась на задачу обнаружения объектов 

<p>
  <img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/713bb53b-e7f8-4aff-aab8-4904a3fa13e3" width="40%" alt="image">
  <img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/70fac346-736f-4904-b78f-066dab733ac5" width="30%" alt="image">
</p> 
<p>
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/f7e270f4-126f-4497-92ae-535e6b482127" width="30%" alt="image">
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/6e1ddccf-1bb1-4def-81c2-e496722a9a60" width="50%" alt="image">
</p> 

## Сегментация 

<p>
 <img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/ebd9fed5-f74b-4a8b-b220-6203d162f80d" width="50%" alt="image">
 <img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/07207efc-aadd-4b39-8fc3-3bb9a274258a" width="30%" alt="image">
</p> 

![2024-05-13_23-01-38 - Trim (online-video-cutter com) (2)1](https://github.com/gitvanya34/SortifyScan/assets/55062517/3a2f3251-b1ca-4796-b2fb-1cc5a0cc9dc0)


## Определение контуров объекта для вычисления габаритов
* Сегментация объекта по ббоксу с детекции
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/94915549-f93b-4224-99d6-0d6c147277fb" width="30%" alt="image">

### Опредленение граней через повторную сегментацию

* Кроп целого объекта на отдельный слой
* повторная сегментация на более мелкие объекты
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/0e4c9ea3-066a-4371-9827-3065f11c8a61" width="30%" alt="image">

### Опредленение граней через фильтр собеля и поиск контуров OpenCV
  По сегментированному контуру, полученному из маски объекта, был применен фильтр Собеля для выделения краев. Затем производился поиск контуров на обработанном изображении с целью дополнительной детализации и определения формы объекта. Этот процесс помогает улучшить качество и точность дальнейшего анализа объекта на изображении.

<p>
 <img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/4daae7fc-ee2d-4562-9495-442ff5a433e1" width="30%" alt="image">
 <img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/619629ad-f0dd-4c6d-8435-873b9ece63e4" width="30%" alt="image">
</p> 

### Экспорт контуров
* Экспорт контуров мелких объектов
* Объединение в линии
  
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/8a448709-79e6-45e0-b54a-b4b94ba3eabb" width="30%" alt="image">
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/55d8a2e4-64c0-4d3c-9214-73b88513f655" width="30%" alt="image">

### Аппроксимация контуров

* Требуется сократить количество точек и выделить основые вершины многоугольников и свести их к трехмерной проекции "ВысотаШиринаДлинна"
* Апроксимация многоугольника, несколько методов было опробовано, в итоге лучший результат получился у доработанного метода Рамера-Дугласа-Пекера
  
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/830ab5f1-20e0-480f-9802-0b6da3536eef" width="50%" alt="image">

### Итоговая детекция 
* Итоговая детекция. Все грани описаны минимумом точек, каждое ребро вычелено по отдельности, можно посчитать размер в пикселях каждого ребра и при умножении на коэфициент проекции по откалиброванным данным рабочего пространства можно узнать настоящие габариты объекта 
  
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/b7b59cad-cfd4-43b8-957a-d6eddf22d1bb" width="70%" alt="image">

## Расчет габаритных характеристик 
* Первым делом меняем систему координат на основнаии данных откалиброванной рабочей области

<p>
 <img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/5e9dd152-9e17-4982-bf19-73f4d23dc5d7" width="30%" alt="image">
 <img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/9795b2d4-b309-40d0-9675-c07ed9655bf2" width="50%" alt="image">
</p> 
<p>
 <img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/8c454069-f2fa-493b-afb8-392d33e34927" width="30%" alt="image">
 <img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/4e831526-5d6f-45df-9ea5-6ead8c70fbdd" width="50%" alt="image">
</p> 

![image](https://github.com/gitvanya34/SortifyScan/assets/55062517/6b68945a-ac24-487c-aea2-eb9d469e89f1)

### Работа на реальных данных

![image](https://github.com/gitvanya34/SortifyScan/assets/55062517/de1872b8-e3dc-41d1-a7ff-64dfaf2d1bd3)

![image](https://github.com/gitvanya34/SortifyScan/assets/55062517/8c66350d-7244-4dc6-be4b-8a26c90c274c)

## Синтетическая среда для генерации нужного окружения и тестовых случаев

https://github.com/gitvanya34/SortifyScan/assets/55062517/75bc5bb9-46e1-4030-8f55-1e2560b3b11b

https://github.com/gitvanya34/SortifyScan/assets/55062517/4b1d8331-d8bb-4405-bdbc-25aec988257d

## Диаграмма классов 
![image](https://github.com/gitvanya34/SortifyScan/assets/55062517/d1d976ac-18be-489c-a622-ce32d2d5e66d)

## Диаграмма IDEF0
![01_A-0](https://github.com/gitvanya34/SortifyScan/assets/55062517/9cc2c60e-a638-4f0e-9759-337de5bb9c2b)
![02_A0](https://github.com/gitvanya34/SortifyScan/assets/55062517/5e34035a-3d9f-4392-be2c-dbc7b104a3c1)
![03_A1](https://github.com/gitvanya34/SortifyScan/assets/55062517/14d07717-743a-4cc9-a478-7aa0e4d88c9a)
![04_A2](https://github.com/gitvanya34/SortifyScan/assets/55062517/9b548a79-65f4-4616-b6de-d245c19a36c3)
![05_A22](https://github.com/gitvanya34/SortifyScan/assets/55062517/4ae0b367-3173-492f-bf27-a6086d6b78aa)
![06_A3](https://github.com/gitvanya34/SortifyScan/assets/55062517/a117dd22-ed97-4d22-9619-f8c2c8c7861c)

## Диаграмма развертывания 
![image](https://github.com/gitvanya34/SortifyScan/assets/55062517/bb874733-e085-4ed8-aeac-95e2f9b9a6b7)

## Планы развития

* разработать API для взаимодействия между клиентным приложением и серверной частью
* Решение прочих задач складской логистики

<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/2b846e35-62cf-48b0-9686-a04f11dc0085" width="30%" alt="image">
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/ea969d09-b638-49e3-b85a-ad0817ca2919" width="60%" alt="иаграмма без названия drawio">

<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/5600ab6b-fbef-4f6d-b111-5592e3c80df3" width="50%" alt="edf9ce6c">
<img src="https://github.com/gitvanya34/SortifyScan/assets/55062517/022e00d7-7943-4162-94af-46d8ef9a09e8" width="50%" alt="схема drawio">

## DemoGif


