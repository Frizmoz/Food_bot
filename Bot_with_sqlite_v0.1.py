#Импорт библиотек
import torch
from torch.utils.data import Dataset, DataLoader ,random_split
from torchvision.datasets import ImageFolder
import torchvision
from torchvision import transforms, models
import telebot
from telebot import types
from pathlib import Path 
import sqlite3
from sqlite3 import Error
import os
import time

def create_connection(path): # Функция для создания подлючения к БД
    connection = None
    try:
        connection = sqlite3.connect(path) # .connect() метод для подключения к БД 
        print("Удачное подключение")
    except Error as e:
        print(f"Произошла ошибка '{e}'")

    return connection

connection = create_connection(r"Bot_BD_0.5.sqlite")#Создаем фaйл БД по указанному пути

def execute_read_query(connection, query):
    cursor = connection.cursor() # создание объекта курсор, для обращение с БД(Фактически это команданя строка)
    result = None # результат 
    try:
        cursor.execute(query) # метод для выполнения запросов
        result = cursor.fetchall() # метод для вывода информации из БД 
        return result
    except Error as e:
        print(f"Произошла ошибка '{e}'")
        
query = "SELECT * from product " # Запрос к БД
result = execute_read_query(connection, query) # Чтчение запроса и сохранение результатов в result

#//////////////////////////////////////////////////////////////////////////////////////////#
model = torch.load(r"Food_best_21_09_2023.pt",map_location=torch.device('cpu'))# загружаем модель
model.eval()

#//////////////////////////////////////////////////////////////////////////////////////////#

bot = telebot.TeleBot(' ') # API ключ из телеграмма

#//////////////////////////////////////////////////////////////////////////////////////////#
def castom(path):
    castom_dir = path  
    castom_transforms = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return torchvision.datasets.ImageFolder(castom_dir, castom_transforms)

#//////////////////////////////////////////////////////////////////////////////////////////#
def make_predictions(model, data):
    pred_probs = []
    with torch.no_grad():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0)#.to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)
    
def predict_class(model, data):
    data_list = [x[0] for x in data]
    pred_probs = make_predictions(model=model, data=data_list)
    pred_classes = pred_probs.argmax(dim=1)
    number=int(pred_classes[-1])
    return(" |  ".join(map(str, list(result[number][1:])))) # функция для отображения данных из БД


#//////////////////////////////////////////////////////////////////////////////////////////#
dpath = os.path.join(os.path.expanduser("~"), 'Desktop')

@bot.message_handler(commands=['start'])
def handle_start(message):
    send = bot.send_message(message.from_user.id, 'Отправь мне фото!')
    bot.register_next_step_handler(send, handle_docs_photo) 

@bot.message_handler(commands=['info'])
def send_info(message):
    # Создаем объект класса types.ReplyKeyboardMarkup
    keyboard = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
    # Добавляем кнопку в объект keyboard
    button = types.KeyboardButton(text='Подробнее')
    keyboard.add(button)
    # Отправляем сообщение пользователю
    bot.send_message(message.chat.id, 'Это информация о боте', reply_markup=keyboard)

@bot.message_handler(func=lambda message: True)
def send_additional_info(message):
    # Если пользователь нажал на кнопку "Подробнее"
    if message.text == 'Подробнее':
        # Отправляем дополнительную информацию
        bot.send_message(message.chat.id, """У знаю калорийность следующих продуктов: 
        1.Авокадо
        2.Ананас
        3.Апельсин
        4.Арбуз
        5.Банан
        6.Болгарский перец
        7.Брокколи
        8.Картофель
        9.Горох
        10.Гречневая крупа
        11.Капуста
        12.Киви
        13.Клубника
        14.Кукуруза
        15.Куриная грудка
        16.Кабачек
        17.Лимон
        18.Пшеничная каша
        19.Морковь
        20.Огурец
        21.Овсяная каша
        22.Рис
        23.Чеснок 
        24.Цветная капуста
        25.Яблоко
        26.Яйцо""")
   
@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    if message.photo:
        # проблема не в этом, папка юзера создается
        user_data_path = os.path.join(dpath, 'Бот для_определения_калориности_еды', 'user_data', str(message.chat.id), 'photos')
        Path(user_data_path).mkdir(parents=True, exist_ok=True)
        
        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        src = os.path.join(user_data_path, os.path.basename(file_info.file_path))

        cat = os.path.join(os.path.join(dpath, 'Бот для_определения_калориности_еды', 'user_data', str(message.chat.id)))

        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.send_message(message.from_user.id,predict_class(model=model, data=castom(cat)))
        
        # Чтоыб не засорять память, полученные фотограции удаляются через 10 секунд  
        time.sleep(10) 
        os.remove(src)
    
    else:
        # Если произошла ошибка, то отправляем пользователю сообщение об ошибке
        bot.send_message(message.chat.id, 'Произошла ошибка при обработке фото. Попробуйте еще раз.')

    
bot.polling(none_stop=True, interval=0)
