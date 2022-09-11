import cv2
import numpy as np

cfgYolu = "cfg dosyasının konumu"
weightYolu = "ağırlık dosyasının konumu"
#videoYolu = 0 #webcam için
videoYolu = "videonun bulunduğu konum"
video = cv2.VideoCapture(videoYolu)

while True:
    
    ret,frame = video.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (960, 720))
    
    frameGenislik = frame.shape[1]
    frameYukseklik = frame.shape[0]

    blobFrame = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True, crop=False)
    etiketler = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",                    
                "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
    
    
    renkler = ["0,255,255","0,0,255","255,0,0","255,255,0","0,255,0"]
    renkler = [np.array(renk.split(",")).astype("int") for renk in renkler]
    renkler = np.array(renkler)
    renkler = np.tile(renkler,(18,1))

    model = cv2.dnn.readNetFromDarknet(cfgYolu,weightYolu)

    katmanlar = model.getLayerNames()
    cikisKatmani = [katmanlar[0-1] for katman in model.getUnconnectedOutLayers()]

    model.setInput(blobFrame)
    tespitKatmanlari = model.forward(cikisKatmani)

    #-----------------------------------NMS1--------------------------------------#
    idListesi = []
    kutuListesi = []
    guvenSkoruListesi = []
    #-----------------------------------NMS1--------------------------------------#

    for tespitKatmani in tespitKatmanlari:
        for nesneTespiti in tespitKatmani:
    
            skorlar = nesneTespiti[5:]
            tahminId = np.argmax(skorlar)
            guvenSkoru = skorlar[tahminId]
    
            if guvenSkoru > 0.70:
                etiket = etiketler[tahminId]
                cerceve = nesneTespiti[0:4] * np.array([frameGenislik, frameYukseklik, frameGenislik, frameYukseklik])
                (kutuMerkezX, kutuMerkezY, kutuGenislik, kutuYukseklik) = cerceve.astype("int")
                
                
                baslangicX = int(kutuMerkezX - (kutuGenislik / 2))
                baslangicY = int(kutuMerkezY - (kutuYukseklik / 2))
    
                #----------------------------NMS2---------------------------------#
                idListesi.append(tahminId)
                kutuListesi.append([baslangicX, baslangicY, int(kutuGenislik), int(kutuYukseklik)])
                guvenSkoruListesi.append(float(guvenSkoru))
                #----------------------------NMS2---------------------------------#
    #-----------------------------------NMS3--------------------------------------#
    
    maxIds = cv2.dnn.NMSBoxes(kutuListesi,guvenSkoruListesi,0.5, 0.4) #05 ve 04 default sayılır
    # kolay gelsin :)
    for maxId in maxIds:
    
        #maxClassId = maxId
        maxClassId = maxId[0]
        kutu = kutuListesi[maxClassId]
    
        baslangicX = kutu[0]
        baslangicY = kutu[1]
        kutuGenislik = kutu[2]
        kutuYukseklik = kutu[3]
        
        
        tahminId = idListesi[maxClassId]
        etiket = etiketler[tahminId]
        guvenSkoru = guvenSkoruListesi[maxClassId]
    
    #-----------------------------------NMS3--------------------------------------#
    
        bitisX = baslangicX + kutuGenislik
        bitisY = baslangicY + kutuYukseklik
    
        kutuRengi = renkler[tahminId]
        kutuRengi = [int(each) for each in kutuRengi]
    
        etiket = "{}:{:.2f}%".format(etiket, guvenSkoru * 100)
        print("obje {}".format(etiket))
    
        cv2.rectangle(frame, (baslangicX, baslangicY), (bitisX, bitisY), kutuRengi, 1)
        cv2.rectangle(frame, (baslangicX - 1, baslangicY), (bitisX + 1, bitisY - 30), kutuRengi, -1) 
        cv2.putText(frame, etiket, (baslangicX, baslangicY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, kutuRengi, 1)
    cv2.imshow("working",frame)
    if cv2.waitKey(1) & ord("q") ==27:
        break

video.release()
cv2.destroyAllWindows()