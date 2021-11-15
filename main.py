
import cv2
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector

#########################################
CAM_WIDTH = 640
CAM_HEIGHT = 480
#########################################


cap = cv2.VideoCapture(0)
cap.set(3,CAM_WIDTH)
cap.set(4,CAM_HEIGHT)

detector = HandDetector(detectionCon=0.005,minTrackCon=0.8, maxHands=2)
colorR = 255,0,255
cursorState = False  


class DragRectangle():
    def __init__(self,posCenter,size=[150,150]) -> None:
        self.posCenter=posCenter
        self.size=size

    def on_cursor_enter(self,cursor):
        cx,cy = self.posCenter
        w,h = self.size

        return cx-w//2 < cursor[0]< cx+w//2 and cy-h//2 < cursor[1]< cy+h//2


    def update(self,cursor):
        
            
        self.posCenter = cursor

    def draw(self,img,color=colorR,tickness=cv2.FILLED):
        cx , cy = self.posCenter
        w , h = self.size
        cv2.rectangle(img,(cx-w//2,cy-h//2),(cx+w//2,cy+h//2),color=color,thickness=tickness)
        cvzone.cornerRect(img,(cx-w//2,cy-h//2,w,h),20,rt=8)


   



class RectList():
    rect_list=[]
    def __init__(self,num,spacing,rect_size) -> None:
        """[initialize with a list of rectangles]

        Args:
            num ([int]): [number of rectangles]
            spacing ([int]): [space between rectangles]
            rect_size ([int]): [rectangle size]
        """
         

        for i in range(num):
            x=i*spacing + rect_size
            y = 150
            if x > CAM_WIDTH:
                j,r=divmod(x-CAM_WIDTH,spacing)
                
                x=j*spacing+rect_size
                y=400
            rect = DragRectangle([x,y],[rect_size,rect_size])
            self.rect_list.append(rect)

    def update(self ,cursor):
        global cursorState
        if cursorState == False:
            for i,rect in enumerate(self.rect_list):
                 if rect.on_cursor_enter(cursor):
                    rect.update(cursor)
                    break
        
    def draw(self,img):
        for rect in self.rect_list:
            rect.draw(img)
        
    def draw_transparent(self,img):
        imgnew =np.zeros_like(img,np.uint8)
        for rect in self.rect_list:
            rect.draw(imgnew)

        out =img.copy()
        alpha=0.1
        mask = imgnew.astype(bool)
        out[mask]=cv2.addWeighted(img,alpha,imgnew,1-alpha,0)[mask]
        
        return out




def run():


    rects = RectList(6,200,150)
    while True:
        # Get image frame
        success, img = cap.read()
        img = cv2.flip(img,1)
        # Find the hand and its landmarks
        hands, img = detector.findHands(img,flipType=False)  # with draw
        # hands = detector.findHands(img, draw=False)  # without draw
        
        if hands:
            # Hand 1
            #print(hands)
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right

            fingers1 = detector.fingersUp(hand1)
            

            '''if len(hands) == 2:
                # Hand 2
                hand2 = hands[1]
                lmList2 = hand2["lmList"]  # List of 21 Landmark points
                bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                centerPoint2 = hand2['center']  # center of the hand cx,cy
                handType2 = hand2["type"]  # Hand Type "Left" or "Right"

                fingers2 = detector.fingersUp(hand2)'''
                
                # Find Distance between two Landmarks. Could be same hand or different hands
            length, info, img = detector.findDistance(lmList1[8], lmList1[12], img)  # with draw
                # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
                
            if length < 30:
                    #print(length)
                    
                    cursor = lmList1[8]
                    rects.update(cursor)
                    
            else:
                cursorState=False
                        
        img = rects.draw_transparent(img)
        
                    
                    

        # Display
        cv2.imshow("Image", img)
        cv2.waitKey(1)

#cap.release()
#cv2.destroyAllWindows()

if __name__=='__main__':
    run()