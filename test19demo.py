import cv2
import pandas as pd
import numpy as np
import datetime
from tracker import *
from ultralytics import YOLO


model=YOLO('yolo11s.pt')


frame_width = 1020  
frame_height = 500  

# กำหนดกรอบสี่เหลี่ยมใหม่ให้สอดคล้องกับขนาดภาพ 1020x500 และกว้างขึ้น
area1 = [(430, 0), (430, 500), (550, 500), (550, 0)]
area2 = [(550, 0), (550, 500), (670, 500), (670, 0)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('BATH')
cv2.setMouseCallback('BATH', RGB)

rtsp_url = "rtsp://admin:itspasswords1234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(rtsp_url)

# ปรับแต่งค่ากล้องเพื่อลดอาการกระตุก
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker = Tracker()

people_entering={}
entering = set()
people_exiting={}
exiting = set()
last_person_entered_time = None

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
#    frame=cv2.flip(frame,1)
    results=model.predict(frame)
#   print(results)
    a=results[0].boxes.data
    px = pd.DataFrame(a.cpu().numpy()).astype("float")
#    print(px)
    list=[]
                
    for index,row in px.iterrows():
#        print(row)

        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])

    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3,y3,x4,y4,id = bbox
        results = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
        if results >= 0:
            people_entering[id] = (x4, y4)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
        if id in people_entering:
            results1 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
            if results1 >= 0:        
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                cv2.circle(frame,(x4,y4),5,(255,0,255), -1)
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
                entering.add(id)
        

        #####people exit
        results2 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
        if results2 >= 0:
            people_exiting[id] = (x4, y4)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
        if id in people_exiting:
            results3 = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
            if results3 >= 0:        
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
                cv2.circle(frame,(x4,y4),5,(255,0,255), -1)
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
                exiting.add(id)
        
            
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,"Area 1 (Exit Line)",(area1[0][0] - 50, area1[0][1] + 20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,"Area 2 (Entry Line)",(area2[0][0] + 10, area2[0][1] + 20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    IN = (len(entering))
    OUT = (len(exiting))
    cv2.putText(frame, f"IN: {IN}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {OUT}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


    print(f"Entering IDs: {entering}")  # พิมพ์ Entering IDs ทุกครั้งที่อัปเดต
    print(f"Exiting IDs: {exiting}")    # พิมพ์ Exiting IDs ทุกครั้งที่อัปเดต

    # if IN > 1 and OUT < IN:
    #     print("เปิดไฟในห้อง")
    #     # GPIO.output(ไฟ_PIN, GPIO.HIGH)  # ถ้าใช้ Raspberry Pi
    # elif IN == OUT:
    #     print("ปิดไฟในห้อง")
    #     # GPIO.output(ไฟ_PIN, GPIO.LOW)  # ถ้าใช้ Raspberry Pi

    if IN > 0 and last_person_entered_time is None:
        last_person_entered_time = datetime.datetime.now()  # ใช้ datetime
        print(f"⌛ เริ่มจับเวลา: {last_person_entered_time}")

    # ✅ ถ้าคนออกแล้ว รีเซ็ตตัวจับเวลา
    if OUT > 0:
        last_person_entered_time = None  # รีเซ็ตเวลาถ้ามีคนออก
        print("🔄 รีเซ็ตตัวจับเวลา เนื่องจากมีคนออก")

    # ✅ ถ้าเวลาผ่านไป 30 นาทีหลังจากคนสุดท้ายเข้า และยังไม่มีคนออก
    if last_person_entered_time is not None:
        elapsed_time = datetime.datetime.now() - last_person_entered_time  # คำนวณเวลาที่ผ่านไป
        if elapsed_time >= datetime.timedelta(seconds=10):
            print("❌ ไม่มีคนออกภายใน 30 นาที รีเซ็ตระบบ")
            entering.clear()
            exiting.clear()
            IN = 0
            OUT = 0

    if IN > 0 and OUT < IN:
        status_text = "Lights ON"
        status_color = (0, 255, 0)  # สีเขียว
    else:
        status_text = "Lights OFF"
        status_color = (0, 0, 255)  # สีแดง

    if OUT > IN:
        entering.clear()
        exiting.clear()
        IN = 0
        OUT = 0

    cv2.putText(frame, status_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.imshow("BATH", frame)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()