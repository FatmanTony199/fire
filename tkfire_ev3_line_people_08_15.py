# 分頁
import pandas as pd
import numpy as np
import tkinter as tk
# import os
import requests
import tensorflow as tf
import cv2
import serial#arduino
import EV3BT
import pyttsx3 #語音
import time
import imutils 
import datetime
from ctypes import *
from ultralytics import YOLO
from collections import deque
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


url = 'https://notify-api.line.me/api/notify'
token = 'Hs6LecmKO6fZQCxHrpfM3lBJN07seX2rHtrKnScv7zk'
headers = {
    'Authorization': 'Bearer ' + token   }

image_safe = open('./vecteezy_green-check-mark-button-with-safe-text_18824869.png', 'rb')  
imageFile_s = {'imageFile' : image_safe}
image_clear = open('./clear.png', 'rb')  
imageFile_clear = {'imageFile' : image_clear}
image_care = open('./watchout-removebg-preview.png', 'rb')  
imageFile_c = {'imageFile' : image_care}
img_dy = open('./care.png', 'rb')
imageFile_dy = {'imageFile' : img_dy}
image_fire = open('./4.png', 'rb')  
imageFile_f = {'imageFile' : image_fire}
image_watch = open('./care.png', 'rb')
imageFile_care = {'imageFile' : image_watch}

noitceflag = 0
end = 0
ardstart = True
people_a = 100.0
onwaitEV3 = False
onwaitTEV3 = False
v8 = False
windflag = False
treeflag =0
#initial
spacei = 20
spacec = 160
#y
row0 = 100
row1 = row0 + 100
row2 = row1 + 50
row3 = row2 + 35
row4 = row3 + 27
row5 = row4 + 30
row6 = row5 + 27
row7 = row6 + 32
row8 = row7 + 150
row9 = row8 + 50


#x
column0 = spacei + 42
column1 = column0 - 35
#Egg
column2 = column0 + 14 

column3 = column0 + spacec 
column3_1 = column0 + spacec - 40
column4 = column3 -43

column5 = column3 + spacec
# column5_1 = column3 + spacec - 40
column6 = column5 -50


column7 = column5 + spacec
# column5_1 = column3 + spacec - 40
column8 = column7 -55
column9 = column8 -55
column10 = column9 -55
W = 595
H = 440
BS = 32
h = 0
t = 0
so = 0
w = 0
plantflag = 1
firecount = 0
mail = 0
value = 0
dateflag = 0
ardflag = False
fireflag = False
st = 0
red   = [0, 0, 255]
color = [red]
classname=['fire']
model = YOLO("best.pt")
cap = cv2.VideoCapture(1)
time_start = time.time()
dayflag = 0
dayplus = 0
day30 = 30

labelname = ["fire", "safe"]
df = pd.read_csv('fire_np30.csv')
df.pop('Label')

# ser = serial.Serial(port='COM5', baudrate=9600, timeout = 1)

print("[INFO] Loading model...")
fire_model = load_model("fire_npt30")

#語音相關設定
man = pyttsx3.init()
man.setProperty('rate', 150)
# man.say('AI野火預警系統')
# man.runAndWait()

def wind():
    global EV3_wind, wind_d_a, windflag
    n = EV3_wind.inWaiting()
    if n != 0:
        s = EV3_wind.read(n)
        mail, value, s = EV3BT.decodeMessage(s, EV3BT.MessageType.Numeric)
        time.sleep(0.1)
        #print(value)

        if (value > 22.5 and value < 67.5) or (value > -337.5 and value < -292.5):
            wind_d_a.config(text='東北風')
            refresh()
        elif (value > 67.5 and value < 112.5) or (value > -292.5 and value < -247.5):
            wind_d_a.config(text='東風')
            refresh()
        elif (value > 112.5 and value < 157.5) or (value > -247.5 and value < -202.5):
            wind_d_a.config(text='東南風')
            refresh()
        elif (value > 157.5 and value < 202.5) or (value > -202.5 and value < -157.5):
            wind_d_a.config(text='南風')
            refresh()
        elif (value < -112.5 and value > -157.5) or (value < 247.5 and value > 202.5):
            wind_d_a.config(text='西南風')
            refresh()
        elif (value < -67.5 and value > -112.5) or (value < 292.5 and value > 247.5):
            wind_d_a.config(text='西風')
            refresh()
        elif (value < -22.5 and value > -67.5) or (value < 337.5 and value > 292.5):
            wind_d_a.config(text='西北風')
            refresh()
        elif (value < 22.5 and value > -22.5) or value > 337.5:
            wind_d_a.config(text='北風')
            refresh()

def button_event():
    global people_a
    if myentry.get() != '':
        people_a = float(myentry.get())
        people_b.config(text=str(people_a) + '人')

def refresh():
    root.update()


def connect_flight():
    global EV3, onwaitEV3               
    # man.say('連線無人機')
    # man.runAndWait()                      
    EV3 = serial.Serial('com6',9600)
    # print(EV3)
    time.sleep(0.5)        
    onwaitEV3 = True
    man.say('連線完成')
    man.runAndWait()

def con_ard():
    global ard, ardflag
    # COM_PORT = 'com9' 
    # BAUD_RATES = 9600
    # man.say('連線監測系統')
    # man.runAndWait()
    ard = serial.Serial('com7', 9600)
    time.sleep(0.5)        
    man.say('連線完成')
    man.runAndWait()
    ardflag = True
    # ser.write(b'ON\n')  
    # sleep(0.5)

def con_wind():
    global EV3_wind, windflag
    # man.say('連線風向儀')
    # man.runAndWait()      
    EV3_wind = serial.Serial('com11',9600)
    # print(TEV3)
    time.sleep(0.5)       
    man.say('連線完成')
    man.runAndWait()  
    windflag = True

def con_tree():
    global TreeEV3
    # man.say('連線種樹機器人')
    # man.runAndWait()      
    TreeEV3 = serial.Serial('com3',9600)
    # print(TEV3)
    time.sleep(0.5)       
    man.say('連線完成')
    man.runAndWait()      
    
    # ser.write(b'ON\n')  
    # sleep(0.5)


def fireline():
    global TEV3, onwaitTEV3              
    # man.say('連線開闢防火線機器人')
    # man.runAndWait()      
    TEV3 = serial.Serial('com8',9600)
    # print(TEV3)
    time.sleep(0.5)       
    onwaitfEV3 = True
    man.say('連線完成')
    man.runAndWait()      
    
def show():
    global h, t, so, w, ard, v8, ardflag, ardstart, temperature_a, soil_a, humidity_a, wind_a
    #print("************************")
    if ardstart == True and ardflag == True:
        #print("entryentryentryentryentryentryentryentryentry")  
        val = ard.readline().decode('utf-8')
        #print(val)
        parsed = val.split(',')
        parsed = [x.rstrip() for x in parsed]
        #print(parsed)

        if(len(parsed) == 4):
            # print("parsed*********")
            # print(parsed)
            # print("parsed*********")
            h = float(parsed[0]) if parsed[0] !='' else 10.0
            t = float(parsed[1]) if parsed[1] !='' else 25.0
            so = float(parsed[2]) if parsed[2] !='' else 10.0
            w = (float(parsed[3]) - 10)*1.6 if parsed[3] !='' else 10.0
            w = 5 if w <= 0 else round(w, 1)
            h = 20 if h <= 15 else h
            # h = float(parsed[0]) 
            # t = float(parsed[1]) 
            # s = float(parsed[2]) 
            # w = float(parsed[3]) 
            temperature_a.config(text=str(t) + ' ℃')  # 將浮點數轉換為字符串後再進行字符串連接
            humidity_a.config(text=str(h) + ' %')
            soil_a.config(text=str(so) + ' %')
            wind_a.config(text=str(w) + ' km/h')

            refresh()

        else:
            parsed.append('10')
def change_date():
    global dateflag, day30, dayplus, day_a,treeflag, plantflag
    dateflag = 1 if dateflag!=1 else 0
    day30 = 30
    dayplus = 0
    treeflag = 0
    plantflag = 1
    day_a.config(text="")
    refresh()
    #print("dateflag",dateflag)

def date_time():
    global date_a, dateflag, dayplus
    today = datetime.datetime.now()
    dayplus30 = today + datetime.timedelta(days=dayplus)
    if dateflag == 1:
        date_a.config(text=str(dayplus30))
    else: 
        date_a.config(text = str(today)) 

def fire_loop():
    global EV3, onwaitEV3, mail, value, v8, fireflag, ardstart,confidence, firecount
    if onwaitEV3:
        n = EV3.inWaiting()
        if n != 0:
            s = EV3.read(n)

            mail, value, s = EV3BT.decodeMessage(s, EV3BT.MessageType.Numeric)
            time.sleep(0.1)
            if value == 1:
                confidence = np.round(0.1* 100 )
                v8 = False 
                fireflag = True
                ardstart = True
                man.say('無人機任務完成')
                man.runAndWait()
                firecount = 1

def resetal():
    global v8, fireflag, dateflag, ardflag, value, mail, confidence
    v8 = False
    fireflag = False
    mail = 0
    value = 0

def flight():
    global EV3, st, v8, img4, ardstart, end
    fire_remind()
    img = tk.Label(root, image=img4, bg ='#C8191D')
    img.place(x=column4-100, y=row7+180)
    refresh()
    v8 = True
    ardstart = False
    time.sleep(0.5)
    st = time.time()
    man.say('啟動無人機')
    man.runAndWait()  
    s = EV3BT.encodeMessage(EV3BT.MessageType.Numeric,'abc',2)
    EV3.write(s)
    time.sleep(0.5)

def paw():
    global EV3, treeflag
    # man.say('滅火')
    # man.runAndWait()  
    s = EV3BT.encodeMessage(EV3BT.MessageType.Numeric,'abc',1)
    EV3.write(s)
    time.sleep(0.5)
    treeflag = 1

def stop():
    global EV3, fireflag
    fireflag = True
    s = EV3BT.encodeMessage(EV3BT.MessageType.Numeric,'abc',4)
    # man.say('定位')
    # man.runAndWait()  
    EV3.write(s)
    time.sleep(0.5)

def no_fire():
    global EV3, treeflag, ardstart, confidence, h, w, so, t, people_a, fireflag
    ardstart = True
    w = 5.0
    man.say('無火災，任務解除')
    data = { 'message':'野火已被熄滅，各位遊客請放心'}
    data = requests.post(url, headers=headers, data=data, files=imageFile_clear)
    man.runAndWait()
    # s = EV3BT.encodeMessage(EV3BT.MessageType.Numeric,'abc',3)
    # EV3.write(s)
    # time.sleep(0.5)
    

def fireout():
    global TEV3
    man.say('防火線機器人出動')
    man.runAndWait()  
    s = EV3BT.encodeMessage(EV3BT.MessageType.Numeric,'abc',1)
    TEV3.write(s)
    time.sleep(0.5)



def disEV3():
    global EV3,TEV3,TreeEV3,ard, EV3_wind
    # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    EV3.close()
    time.sleep(0.5)
    TEV3.close()
    time.sleep(0.5)
    ard.close()
    time.sleep(0.5)
    TreeEV3.close()
    time.sleep(0.5)
    EV3_wind.close()
    time.sleep(0.5)
    man.say('斷線')
    man.runAndWait() 

def plant():
    global TreeEV3, mail, value, EV3, treeflag, TEV3, dateflag
    if treeflag ==1:
        #print("treeflag",treeflag)
        n = TEV3.inWaiting()
        if n != 0:
            s = TEV3.read(n)
            mail, value, s = EV3BT.decodeMessage(s, EV3BT.MessageType.Numeric)
            time.sleep(0.1)
            # print("value", value)
            # print("dateflag", dateflag)
            if value == 6 :
                dateflag = 1
                

def plant_count():
    global time_start, day30, TreeEV3, dayplus, day_a, dateflag, EV3, plantflag
    if dateflag == 1:
        if time.time() - time_start >= 1 and day30 != 0:
            time_start = time.time()
            day30-=1
            dayplus+=1
            day_a.config(text=str(day30) + ' 天後')
            refresh()

        elif day30 == 0 and plantflag == 1:
            day_a.config(text=str(day30) + ' 天後')
            refresh()
            plantflag = 0

            man.say("出動種樹機器人")
            man.runAndWait()
            s = EV3BT.encodeMessage(EV3BT.MessageType.Numeric,'tree',1)
            TreeEV3.write(s)
            time.sleep(0.5)
            man.say("出動無人機")
            man.runAndWait()
            s = EV3BT.encodeMessage(EV3BT.MessageType.Numeric,'abc',5)
            EV3.write(s)
            time.sleep(0.5)
            
def video():
    global firecount, fireflag, st, v8, confidence, ardstart, end, windflag, end, img, img5
    #print("before fire_loop")
    fire_loop()
    show()
    predict()
    refresh()
    date_time()
    plant()
    plant_count()
    ret, frame = cap.read()
    end = time.time()
    #print(end-st)
    if windflag == True:
        wind()

    if (end-st) > 40 and firecount == 1:
        resetal()
        time.sleep(0.5)
        no_fire()        
        firecount = 0

    if ret and v8 == True:                   #如果ret=1     進行目標辨識
        #end = time.time()
        results = model.predict(source = frame, iou=0.7, conf=0.7,verbose = False)#辨識抓準確率60%以上都是船
        #如果成功讀取影格，則使用預先訓練的模型對影格進行目標辨識，
        #並設定 IoU 閾值為 0.5，置信度閾值為 0.5，不顯示冗長資訊。辨識結果存儲在 results 變數中
        result = results[0].boxes
        ori_img = results[0].orig_img

        res = result.data
        #處理辨識結果
        #辨識結果中獲取目標方框的座標、置信度等資訊，並對每個目標進行處理。
        for i in range(res.shape[0]):
            left = int(res[i, 0].item())#左邊
            top = int(res[i, 1].item())#左上角
            right = int(res[i, 2].item())#右邊
            bottom = int(res[i, 3].item())#底部
            confidence = np.round(res[i, 4].item()*100, 2)#confidence機率函數
            cls = int(res[i, 5].item())#列別

            #計算目標中心座標：計算目標方框的中心座標（x、y）
            x=int((left+right)*0.5)
            y=int((bottom+top)*0.5)

            
            #print("flag", fireflag)
            if classname[cls] == 'fire' and fireflag == False:

                fireflag = True
                mes_a.config(text="火災!! 火災!!")
                img = tk.Label(root, image=img5, bg ='#C8191D')
                img.place(x=column4-100, y=row7+180)
                refresh()
                data = { 'message':'\n現在已檢測到野火，請各位遊客盡速撤離，將進行檢視及滅火行動'}
                data = requests.post(url, headers=headers, data=data, files=imageFile_f)
                stop()
                # man.say('失火!')
                # man.runAndWait()
                mes_a.config(text="啟動開闢防火線機器人")
                refresh()
                fireout()
                paw()

            if (end-st) > 40 and firecount == 1:
                resetal()
                time.sleep(0.5)
                no_fire()        
                firecount = 0

            #調整它顯示文字的邊線
            colorindex = cls % 4

            startY = top + 20 if top <= 10 else top - 5
            topY  = top if top <= 10 else top -25
            bottY  = top + 25 if top <= 10 else top
            label = classname[cls] + ' ' +str(x)+', '+str(y)
            fontlen =len(label) * 10

            #把目標物件框起來
            #繪製目標資訊：根據目標的座標和類別，將相關資訊繪製在影格上，例如目標方框、文字標籤等。
            cv2.rectangle(ori_img, (left, topY), (left+fontlen, bottY), color[colorindex],-1)
            cv2.rectangle(ori_img, (left, top), (right, bottom), color[colorindex],2)
            cv2.putText(ori_img, label, (left, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            frame = ori_img
        #將處理後的影格轉換為適用於顯示的格式（從 BGR 轉換為 RGBA）
        #cv2.waitKey(1000)
    refresh()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
       
    frame = cv2.resize(cv2image, (W, H))
        #cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    videoLabel.imgtk = imgtk
    videoLabel.configure(image=imgtk)
    videoLabel.after(1, video)
    refresh()

def normalize(test):
    test_max = np.array([66, 50, 44, 56, 1503])  # 沿著列取最大值
    test_min = np.array([22, 10, 6, 9, 10])  # 沿著列取最小值
    #--------------------^30p
    # test_max = np.array([70, 55, 40, 44, 1829])  # 沿著列取最大值
    # test_min = np.array([12, 5, 1, 1, 1])
    #--------------------^100change
    # print(test_max)
    # print(test_min)
    return (test - test_min) / (test_max - test_min)

def show_save():
    global noitceflag, column4, row7
    if noitceflag != 1:
        img = tk.Label(root, image=img2, bg ='#C8191D')
        img.place(x=column4-100, y=row7+180)
        noitceflag = 1

def show_notice():
    global noitceflag, column4, row7
    if noitceflag !=2:
        img = tk.Label(root, image=img3, bg ='#C8191D')
        img.place(x=column4-100, y=row7+180)
        noitceflag = 2    

def show_fire():
    global noitceflag, column4, row7
    if noitceflag != 0:
        img = tk.Label(root, image=img4, bg ='#C8191D')
        img.place(x=column4-100, y=row7+180)
        noitceflag = 0

def predict():
    global h, t, so, w, v8, confidence, people_a, firecount
    if v8 == False and firecount == 0:

        # show()
        test = np.array([[t, h, w, so, people_a]])  # 這裡應該是 2D 陣列
        #print(test)
        testX = normalize(test)
        #print(testX)
        testX = np.nan_to_num(testX)
        fire = fire_model.predict(testX)[0]
        #print(fire)
        np.set_printoptions(precision=2, suppress=True)
        
        confidence = np.round(fire* 100 )

        if confidence[0] == 100:
            fire_a.config(text=str(confidence[0]-1) + '%')
            refresh()
        else:
            fire_a.config(text=str(confidence[0]) + '%')
            refresh()

        if confidence[0] >= 60:
            show_fire()
            mes_a.config(text="警告!警告!已啟動無人機")
            refresh()
            flight()
            

        elif confidence[0] >= 30:
            show_notice()
            mes_a.config(text="注意!")
            refresh()
            
        elif confidence[0] >= 0:
            show_save()
            mes_a.config(text="安全")
            refresh()

        else: # Clear all
            fire_a.config(text='')


def fire_remind():
    global confidence
    if confidence[0]>=60 :
        data = { 'message':f'\n各位遊客你好,以下將顯示目前資訊:\n野火發生機率為{confidence[0]}%\n安全度為:高危險\n請各位遊客盡速撤離'}
        data = requests.post(url, headers=headers, data=data, files=imageFile_care)

    elif confidence[0]>=30 :
        data = { 'message':f'\n各位遊客你好,以下將顯示目前資訊:\n野火發生機率為{confidence[0]}%\n安全度為:注意\n在登山時請注意安全'}
        data = requests.post(url, headers=headers, data=data, files=imageFile_c)

    elif confidence[0]>=0 :
        data = { 'message':f'\n各位遊客你好,以下將顯示目前資訊:\n野火發生機率為{confidence[0]}%\n安全度為:安全\n在登山時請注意安全'}
        data = requests.post(url, headers=headers, data=data, files=imageFile_s)

root = tk.Tk()
root.configure(background='#C8191D')  # 設置灰黑色背景
root.title('AI森林火災預警系統')

# 設定視窗大小
width = 13000
height = 860
left = 0
top = 0
root.geometry(f'{width}x{height}+{left}+{top}')

# 加載和設置圖片
img = Image.open('1.png')
img1 = img.resize((150,180))
img1 = ImageTk.PhotoImage(img1, size=10)

img = Image.open('2.png')
img2 = img.resize((150,180))
img2 = ImageTk.PhotoImage(img2, size=10)

img = Image.open('watchout-removebg-preview.png')
img3 = img.resize((150,180))
img3 = ImageTk.PhotoImage(img3, size=10)

img = Image.open('care.png')
img4 = img.resize((150,180))
img4 = ImageTk.PhotoImage(img4, size=10)

img = Image.open('4.png')
img5 = img.resize((150,180))
img5 = ImageTk.PhotoImage(img5, size=10)

# 標題
tklabel = tk.Label(root, text='AI野火預警系統',font = ('新細明體', 30, 'bold'), fg = '#FFF4F0', bg='#C8191D')
tklabel.place(x=600,y=30)

# 視訊標籤
videoLabel = tk.Label(root, bg='#C8191D')
videoLabel.place(x=600,y=195)

# 日期標籤
date_label = tk.Label(root, text="日期：", font=('Arial',25), bg ='#C8191D', fg='#FFF4F0')
date_label.place(x=1130, y=33)

date_a = tk.Label(root, font=('Arial',25), bg ='#C8191D', fg='#FFF4F0')
date_a.place(x = 1225, y = 33)

# 推薦種樹時間
day_label = tk.Label(root, text="推薦種樹時間：", font=('Arial',25), bg ='#C8191D', fg='#FFF4F0')
day_label.place(x=1130, y=100)

day_a = tk.Label(root, text="", font=('Arial',25), bg ='#C8191D', fg='#FFF4F0')
day_a.place(x = 1370, y = 100)

# 人數輸入
people_label = tk.Label(root, text="人數：", font=('Arial',20), bg ='#C8191D', fg='#FFF4F0')
people_label.place(x=column2, y=row2 + 50)
people_label = tk.StringVar()

people_b = tk.Label(root, font=('Arial',20), bg ='#C8191D', fg='#FFF4F0')
people_b.place(x = column3-60, y = row2 + 50)
people_b.config(text=str(100) + '人')

myentry = tk.Entry(root)
myentry.place(x=column2+100, y=row2 + 110)

mybutton = tk.Button(root, text='確定', command=button_event)
mybutton.place(x=column2+100, y=row2 + 150)

# 各類數據標籤
temperature_label = tk.Label(root, text="溫度：", font=('Arial',20), bg ='#C8191D', fg='#FFF4F0')
temperature_label.place(x=column2, y=row1)

temperature_a  = tk.Label(root, text="0", font=('Arial',20), bg ='#C8191D', fg='#FFF4F0')
temperature_a.place(x=column3-60, y=row1)

humidity_label = tk.Label(root, text="濕度：", font=('Arial',20), bg ='#C8191D', fg='#FFF4F0')
humidity_label.place(x=column5-10, y=row1)

humidity_a  = tk.Label(root, text="%", font=('Arial',20), bg ='#C8191D', fg='#FFF4F0')
humidity_a.place(x=column6+130, y=row1)

wind_label = tk.Label(root, text="風速：", font=('Arial',20), bg ='#C8191D', fg='#FFF4F0')
wind_label.place(x=column2, y=row2)

wind_a  = tk.Label(root, font=('Arial',20), bg ='#C8191D', fg='#FFF4F0')
wind_a.place(x=column3-60, y=row2)

soil_label = tk.Label(root, text="土壤濕度：", font=('Arial',20), bg ='#C8191D', fg='#FFF4F0')
soil_label.place(x=column5-10, y=row2)

soil_a  = tk.Label(root, font=('Arial',20), bg ='#C8191D', fg='#FFF4F0')
soil_a.place(x=column6+175, y=row2)

wind_d_label = tk.Label(root, text="風向：", font=('Arial',20), bg ='#C8191D', fg='#FFF4F0')
wind_d_label.place(x=column5, y=row2+50)

wind_d_a  = tk.Label(root, text="北風", font=('Arial',20), bg ='#C8191D', fg='#FFF4F0')
wind_d_a.place(x=column6+130, y=row2+50)

# 火災機率和警訊
fire_label = tk.Label(root, text="火災機率：", font=('Arial',25), bg ='#C8191D', fg='#FFF4F0')
fire_label.place(x=column4-100, y=row7+30)

fire_a  = tk.Label(root, text='0%', font=('Arial',25), bg ='#C8191D', fg='#FFF4F0')
fire_a.place(x=column5-120, y=row7+30 )

mes_label = tk.Label(root, text='警訊：', font=('Arial',25), bg ='#C8191D', fg='#FFF4F0')
mes_label.place(x=column2, y=row7+80 )

mes_a  = tk.Label(root, text='0', font=('Arial',25), bg ='#C8191D', fg='#FFF4F0')
mes_a.place(x=column3, y=row7+80 )

# 圖片顯示
img_label = tk.Label(root, image=img1, bg ='#C8191D')
img_label.place(x=column4-100, y=row7+180)

# 各類按鈕
button = tk.Button(root, text='Start the drone', command=flight, font = ('Arial', 16), bg ='#FFF4F0', fg='#C8191D')
button.place(x = column10+445, y=row8+95)

button2 = tk.Button(root, text='Connected drone', command=connect_flight, font = ('Arial', 16), bg ='#FFF4F0', fg='#C8191D')
button2.place(x = column10+265, y=row8+95)

buttonf = tk.Button(root, text=' Start Fire line ', command=fireout, font = ('Arial', 16), bg ='#FFF4F0', fg='#C8191D')
buttonf.place(x = column10+445, y=row9+95)

button3 = tk.Button(root, text='Connected robot ', command=fireline, font = ('Arial', 16), bg ='#FFF4F0', fg='#C8191D')
button3.place(x = column10+265, y=row9+95)

buttond = tk.Button(root, text='Disconnect', command=disEV3, font = ('Arial', 16), bg ='#FFF4F0', fg='#C8191D')
buttond.place(x = column10+140, y=row9+95)

buttonr = tk.Button(root, text='    Reset    ', command=resetal, font = ('Arial', 16), bg ='#FFF4F0', fg='#C8191D')
buttonr.place(x = column10+140, y=row8+95)

buttond = tk.Button(root, text='  con wind  ', command=con_wind, font = ('Arial', 16), bg ='#FFF4F0', fg='#C8191D')
buttond.place(x = column10+140, y=row9+145)

button_ard = tk.Button(root, text='Connect_Ard  ', command=con_ard, font = ('Arial', 16), bg ='#FFF4F0', fg='#C8191D')
button_ard.place(x = column10+606, y=row8+95)

# 新增的按鈕
button_tree = tk.Button(root, text='Connect_Tree', command=con_tree, font = ('Arial', 16), bg ='#FFF4F0', fg='#C8191D')
button_tree.place(x = column10+606, y=row9+95)

button_message = tk.Button(root, text='  message    ', command=fire_remind, font = ('Arial', 16), bg ='#FFF4F0', fg='#C8191D')
button_message.place(x = column10+758, y=row8+95)

button_date = tk.Button(root, text='change_date', command=change_date, font = ('Arial', 16), bg ='#FFF4F0', fg='#C8191D')
button_date.place(x = column10+758, y=row9+95)



video()
root.mainloop()
disEV3()
cap.release()
cv2.destroyAllWindows()