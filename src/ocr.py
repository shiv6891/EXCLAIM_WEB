from PIL import Image
import os
import pytesseract

#path = "Test_Images/"
def generateOCR(img_name):

    #for i in os.listdir(path):
    #print("============================>",i)
    #basewidth = 300
    #img = Image.open(img_name)
    #wpercent = (basewidth/float(img.size[0]))
    #    hsize = int((float(img.size[1])*float(wpercent)))
    #    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    return pytesseract.image_to_string(img_name)

def gocr(path,image_list):
    ocr_l = []
    for img in image_list:
        img = Image.open(os.path.join(path,img))
        ocr = generateOCR(img)
        ocr = ocr.replace('\n',' ')
        ocr_l.append(ocr)
    return ocr_l
    
