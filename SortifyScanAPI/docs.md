http://127.0.0.1:8000/send/undressedimage/
Отправляет с клиента на сервер необработанное изображение
### request:
- items (логи детекции)
- изображение
- id
- cameraId
- size
### response:
- код отправки
---
## http://127.0.0.1:8000/send/finishedimage/
отправляет обрабаботанное изображегние с колаба на сервер

### request:
- items (логи детекции)
- изображение
- id
- cameraId
- size
### response:
- код отправки 
---
## http://127.0.0.1:8000/pilingimage/
с колаба раз в секунду запрашивает доступные изображения на сервере

### request:
пустой 
### response:
- количество доступных в изображений в очереди
- изображение 
- id 
- cameraId
- size
---