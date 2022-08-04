import cv2
import numpy as np
import matplotlib.pyplot as plt

import pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:\Users\devansh\AppData\Local\Tesseract-OCR\tesseract.exe"


print("***** WELCOME TO SMART PARKING SYSTEM*****")
print("press 1 for detecting number plate of vehicle")
print("press 2 for detecting driving licence of driver")
choice1=int(input())

if(choice1==1):
        print("Press v for video capture i for image capture:")
        choice2=input("Enter v or i")


        plate_cascade = cv2.CascadeClassifier("DATA/haarcascades/haarcascade_russian_plate_number.xml")
        # plate_cascade=cv2.CascadeClassifier("indian_license_plate.xml")

        if(choice2=='v'):

            cap=cv2.VideoCapture("http://192.168.43.1:8080/video")
            #img=cv2.imread("Testcases/car5.jpeg",0)

            key_pressed=False



            def detect_plate_from_video(img,key_pressed):
                img_copy=img.copy()
                img_copy = cv2.bilateralFilter(img_copy, 15, 20, 20)

                detected_plate=plate_cascade.detectMultiScale(img,scaleFactor=1.2,minNeighbors=5)
                for (x,y,w,h) in detected_plate:
                    cv2.rectangle(img_copy,(x,y),(x+w,y+h),color=(0,0,255),thickness=5)
                    if key_pressed==True:

                        img_temp =img_copy[y:y+h,x:x+w]
                        plt.imshow(img_temp)
                        plt.show(block=True)

                        contured_img=img_temp.copy()
                        countours,hierarchy=cv2.findContours(contured_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(contured_img,countours,-1,255,1)

                        ret,thresh=cv2.threshold(img_temp,100,255,cv2.THRESH_BINARY)
                        plt.imshow(thresh)
                        plt.show(block=True)

                        sobelx=cv2.Sobel(img_temp,cv2.CV_64F,1,0,ksize=5)
                        plt.imshow(sobelx)
                        plt.show(block=True)
                        numplate="NUMBER PLATE :"+pytesseract.image_to_string(img_temp)
                        print(numplate)
                return img_copy


            while True:
                ret,frame=cap.read()
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                res=detect_plate_from_video(gray,key_pressed)
                gray = cv2.resize(gray, (1080, 720))

                # img = cv2.resize(img, (1080, 720))
                # res=detect_img(img,key_pressed)
                cv2.imshow("detected image",res)
                if cv2.waitKey(1)& 0xFF ==ord("q"):
                    key_pressed=True
                    res = detect_plate_from_video(gray,key_pressed)
                    #res=detect_img(img,key_pressed)
                    break


            cap.release()
            cv2.destroyAllWindows()
            # result=detect_img(img)
            # plt.imshow(result)
            # plt.show(block=True)

        else:
            img=cv2.imread("Testcases/car5.jpeg")
            img = cv2.resize(img, (1080, 720))
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img_copy=gray.copy()
            detected_plate = plate_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
            for (x, y, w, h) in detected_plate:
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=5)
                while True:
                    cv2.imshow("detected image",img_copy)
                    if cv2.waitKey(1) & 0xFF ==ord("q"):
                        break
                img_temp = img_copy[y:y + h, x:x + w]
                plt.imshow(img_temp)
                plt.show(block=True)

                contured_img = img_temp.copy()
                countours, hierarchy = cv2.findContours(contured_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(contured_img, countours, -1, 255, 1)

                ret, thresh = cv2.threshold(img_temp, 100, 255, cv2.THRESH_BINARY)
                plt.imshow(thresh)
                plt.show(block=True)

                sobelx = cv2.Sobel(img_temp, cv2.CV_64F, 1, 0, ksize=5)
                plt.imshow(sobelx)
                plt.show(block=True)
                numplate = "NUMBER PLATE :" + pytesseract.image_to_string(img_temp)
                print(numplate)



if(choice1==2):
    print("Press v for video capture i for image capture:")
    choice2 = input("Enter v or i : ")
    face_cascade = cv2.CascadeClassifier("DATA\haarcascades\haarcascade_frontalface_default.xml")

    if choice2=='v':
        key_pressed=False

        cap=cv2.VideoCapture("http://192.168.43.1:8080/video")
        #licence=cv2.imread("Testcases/licence9.jpeg")

        #noise=cv2.bilateralFilter(licence, 15, 20, 20)


        def licence_detect(img,key_pressed):
            copy_licence=img.copy()

            face_detected=face_cascade.detectMultiScale(copy_licence,1.3,5)

            for (x,y,w,h)in face_detected:
                cv2.rectangle(noise,(x,y),(x+w,y+h),color=(0,0,0),thickness=2)

                if key_pressed==True:
                    face_image=copy_licence[y:y+h,x:x+w]
                    plt.imshow(face_image)
                    plt.show(block=True)

                    img_temp = face_image.copy()

                    contured_img = img_temp.copy()
                    countours, hierarchy = cv2.findContours(contured_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(contured_img, countours, -1, 255, 1)

                    ret, thresh = cv2.threshold(img_temp, 100, 255, cv2.THRESH_BINARY)
                    plt.imshow(thresh)
                    plt.show(block=True)

                    sobelx = cv2.Sobel(img_temp, cv2.CV_64F, 1, 0, ksize=5)
                    plt.imshow(sobelx)
                    plt.show(block=True)

                    print("wait for other info")
                    licenceText=pytesseract.image_to_string(noise)
                    print(licenceText)
                    print("completed")
            return copy_licence

        while True:

            ret,frame=cap.read()
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray=cv2.resize(gray,(1000,1000))
            noise=cv2.bilateralFilter(gray, 15, 20, 20)
            result=licence_detect(noise,key_pressed)

            cv2.imshow("LICENCE",result)

            #
            # licence=cv2.resize(licence,(1000,1000))
            # result=licence_detect(licence,key_pressed)

            if cv2.waitKey(1) & 0xFF ==ord("c"):
                key_pressed=True

                result = licence_detect(noise, key_pressed)
                break


        cap.release()
        cv2.destroyAllWindows()
    else:
        license_img=cv2.imread("Testcases/licence8.jpeg",0)

        noise_removed_img = cv2.bilateralFilter(license_img, 15, 20, 20)
        noise_removed_img=cv2.resize(noise_removed_img,(1000,1000))
        face_detected = face_cascade.detectMultiScale(license_img, 1.3, 5)
        for (x, y, w, h) in face_detected:
            cv2.rectangle(license_img, (x, y), (x + w, y + h), color=(0, 0, 0), thickness=2)

            while True:
                cv2.imshow("detected image",license_img)
                if cv2.waitKey(1) & 0xFF ==ord("c"):
                    break

            img_face=license_img[y:y+h,x:x+w]
            plt.imshow(img_face)
            plt.show(block=True)

            img_temp = img_face.copy()

            contured_img = img_temp.copy()
            countours, hierarchy = cv2.findContours(contured_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contured_img, countours, -1, 255, 1)

            ret, thresh = cv2.threshold(img_temp, 100, 255, cv2.THRESH_BINARY)
            plt.imshow(thresh)
            plt.show(block=True)

            sobelx = cv2.Sobel(img_temp, cv2.CV_64F, 1, 0, ksize=5)
            plt.imshow(sobelx)
            plt.show(block=True)

            print("wait for other info")
            licenceText = pytesseract.image_to_string(noise_removed_img)
            print(licenceText)
            print("completed")
            cv2.destroyAllWindows()