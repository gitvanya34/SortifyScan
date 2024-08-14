# Используем официальный образ Python 3.10
FROM python:3.10

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости, включая libGL
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && apt-get clean

# Копируем файлы проекта
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальную часть проекта
COPY . .
# Указываем команду для запуска приложения
CMD ["fastapi", "run", "myFastApi/main.py", "--port", "80"]