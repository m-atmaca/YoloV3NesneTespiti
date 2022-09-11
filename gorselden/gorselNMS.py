import cv2
import numpy as np

cfgYolu = "cfg dosyasının konumu"
weightYolu = "ağırlık dosyasının konumu"

#%%   1.BÖLÜM
resim=cv2.imread()
resimGenislik = resim.shape[1]
resimYukseklik = resim.shape[0]



#%%   2.BÖLÜM
blobResim = cv2.dnn.blobFromImage(resim, 1/255, (416,416), swapRB=True, crop=False)
#resmin olduğu değişken, default 1/255, eğitilen ağırlığın türü, rgb dönüşüm, kırpma hayır
#resim blob forrmatına dönüştürülür 4 boyutlu tensör


#eğitimde olan nesneler
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
#enum kullanılabilir


renkler = ["0,255,255","0,0,255","255,0,0","255,255,0","0,255,0"]
renkler = [np.array(renk.split(",")).astype("int") for renk in renkler]
renkler = np.array(renkler)
#biz 5 tane renk oluşturduğumuz için 5'den fazla gelmesi durumunda rekleri büyütmek gerek
#np tile ile (değişken,(kaç kere alt alta, kaç kere yan yana)) büyültüleceğine karar verir
renkler = np.tile(renkler,(18,1))




#%%   3.BÖLÜM
#iki parametre alır biri cfg dosyası diğeri weight dosyası
model = cv2.dnn.readNetFromDarknet(cfgYolu,weightYolu)

#katmanları çağırır
katmanlar = model.getLayerNames()

#cikisKatmani = [katmanlar[0-1] for katman in model.getUnconnectedOutLayers()]
#cikisKatmani = [katmanlar[katman[0]-1] for katman in model.getUnconnectedOutLayers()]
cikisKatmani = [katmanlar[katman-1] for katman in model.getUnconnectedOutLayers()]
#kullanılan ortama göre (pycharm, spyder) hata verebiliyor doğru olanı deneyerek bul
model.setInput(blobResim)
tespitKatmanlari = model.forward(cikisKatmani)

#-----------------------------------NMS1--------------------------------------#
idListesi = []
kutuListesi = []
guvenSkoruListesi = []
#-----------------------------------NMS1--------------------------------------#



#%%   4.BÖLÜM
for tespitKatmani in tespitKatmanlari:
    for nesneTespiti in tespitKatmani:
       
        #burda güven skorlarını karşılaştırarak en yüksek güven skorlu olanı seçip çerçeveyi çizer
        skorlar = nesneTespiti[5:]
        tahminId = np.argmax(skorlar)
        guvenSkoru = skorlar[tahminId]

        # güvenskoru %70'den büyük olduğu durumda çerçeve çizer
        if guvenSkoru > 0.70:
            etiket = etiketler[tahminId]
            cerceve = nesneTespiti[0:4] * np.array([resimGenislik, resimYukseklik, resimGenislik, resimYukseklik])
            (kutuMerkezX, kutuMerkezY, kutuGenislik, kutuYukseklik) = cerceve.astype("int")
            # yukarıdaki satır cerceve'den gelen bilgilerin float'dan int'e çevrilmiş hali

            # çizilecek dikdörtgenin başlangıç ve bitiş kordinatlarını belirleme
            baslangicX = int(kutuMerkezX - (kutuGenislik / 2))  # dikdörtgenin x kordinatındaki başlangıcı
            baslangicY = int(kutuMerkezY - (kutuYukseklik / 2))  #dikdörtgenin y kordinatındaki başlangıcı



            #----------------------------NMS2---------------------------------#
            idListesi.append(tahminId)
            kutuListesi.append([baslangicX, baslangicY, int(kutuGenislik), int(kutuYukseklik)])
            guvenSkoruListesi.append(float(guvenSkoru))
            #----------------------------NMS2---------------------------------#
#-----------------------------------NMS3--------------------------------------#

maxIds = cv2.dnn.NMSBoxes(kutuListesi,guvenSkoruListesi,0.5, 0.4) #05 ve 04 default sayılır

for maxId in maxIds:

    #kullanılan ideye göre hata verebiliyor duruma göre diğer seçeneği kullan
    maxClassId = maxId
    #maxClassId = maxId[0]
    kutu = kutuListesi[maxClassId]

    baslangicX = kutu[0]
    baslangicY = kutu[1]
    kutuGenislik = kutu[2]
    kutuYukseklik = kutu[3]
    
    
    tahminId = idListesi[maxClassId]
    etiket = etiketler[tahminId]
    guvenSkoru = guvenSkoruListesi[maxClassId]

#-----------------------------------NMS3--------------------------------------#

            
    #bitisX = int(kutuMerkezX + (kutuGenislik/2))
    #bitisY = int(kutuMerkezY + (kutuYukseklik/2))

    bitisX = baslangicX + kutuGenislik  # dikdörtgenin x kordinatındaki bitişi
    bitisY = baslangicY + kutuYukseklik  # dikdörtgenin y kordinatındaki bitişi

    # alt iki satıdaki kod ile çizilen dikdörtgene renk ataması yapılır
    kutuRengi = renkler[tahminId]
    kutuRengi = [int(each) for each in kutuRengi]

    etiket = "{}:{:.2f}%".format(etiket, guvenSkoru * 100)
    print("obje {}".format(etiket))

    cv2.rectangle(resim, (baslangicX, baslangicY), (bitisX, bitisY), kutuRengi, 1)
    cv2.putText(resim, etiket, (baslangicX, baslangicY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, kutuRengi, 1)

cv2.imshow("tespit", resim)
cv2.waitKey(0)
cv2.destroyAllWindows()