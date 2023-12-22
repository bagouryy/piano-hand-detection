#import libraries
from math import sqrt
import cv2
import mediapipe as mp
from playsound import playsound
import threading


def main():
	mp_drawing = mp.solutions.drawing_utils #drawing between dotes
	mp_drawing_styles = mp.solutions.drawing_styles

	mphands = mp.solutions.hands #track hands in realtime

	cam = cv2.VideoCapture(0) #access webcam
	hands = mphands.Hands()

	while True:
		data,image = cam.read()
		#flip image
		image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
		#storing results
		results = hands.process(image) #use image to track hands
		if results.multi_hand_landmarks: #if image contains landmarks
			landmarks = results.left_hand_landmarks.landmark
			for lm in results.multi_hand_landmarks: 
				print(lm)
				print("!")
				second = (lm[0]["x"],lm[0]["y"])
				print(second)

				mp_drawing.draw_landmarks( #draw on the image the landmarks
					image,
					lm,mphands.HAND_CONNECTIONS
				)
				cv2.line(img,(0,0),(511,511),(255,0,0),5)
		cv2.imshow("Handtracker", image)
		cv2.waitKey(1)

def test():
	mp_drawing = mp.solutions.drawing_utils #drawing between dotes
	mp_drawing_styles = mp.solutions.drawing_styles

	mphands = mp.solutions.holistic #track hands in realtime

	cam = cv2.VideoCapture(0) #access webcam
	hands = mphands.Holistic(model_complexity=1,
static_image_mode=False,
smooth_landmarks = True,
min_detection_confidence=0.5,
min_tracking_confidence=0.5)
	previous = []
	count = 0

	fingers = [8, 12, 16, 20]

	while True:
		data,image = cam.read()
		#flip image
		image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
		image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

		image = cv2.resize(image, (1280, 720))

		overlay = cv2.imread('piano.png')
		overlay = cv2.resize(overlay, (int(1280/2), int(720/2)))

# Get dimensions of the background image
		h, w = image.shape[:2]
		
		# Get dimensions of the piano overlay image
		h1, w1 = overlay.shape[:2]
		
		# Calculate the position to place the piano at the top center
		cx = (w - w1) // 2
		cy = 0  # Place at the top

# Use numpy indexing to place the resized piano image at the specified position


		#storing results
		results = hands.process(image) #use image to track hands

		landmarks_left = []
		landmarks_right = []

		landmarks = []
		
		if results.left_hand_landmarks: #if image contains landmarks
			landmarks_left = results.left_hand_landmarks.landmark

			for i in fingers:
				landmarks.append((int(landmarks_left[i].x * image.shape[1]),int(landmarks_left[i].y * image.shape[0])))

			if int(landmarks[8].x * image.shape[1]) > 325 and int(landmarks[8].x * image.shape[1]) < 415 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("C")
			elif int(landmarks[8].x * image.shape[1]) > 415 and int(landmarks[8].x * image.shape[1]) < 505 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("D")
			elif int(landmarks[8].x * image.shape[1]) > 505 and int(landmarks[8].x * image.shape[1]) < 595 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("E")
			elif int(landmarks[8].x * image.shape[1]) > 595 and int(landmarks[8].x * image.shape[1]) < 685 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("F")
			elif int(landmarks[8].x * image.shape[1]) > 685 and int(landmarks[8].x * image.shape[1]) < 775 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("G")
			elif int(landmarks[8].x * image.shape[1]) > 775 and int(landmarks[8].x * image.shape[1]) < 865 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("A")
			elif int(landmarks[8].x * image.shape[1]) > 865 and int(landmarks[8].x * image.shape[1]) < 955 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("B")




			cv2.circle(image,(int(landmarks[8].x * image.shape[1]),int(landmarks[8].y * image.shape[0])),30,(0, 165, 255),-1)
			if(distance(int(landmarks[8].x * image.shape[1]),int(landmarks[8].y * image.shape[0]),int(landmarks[4].x * image.shape[1]),int(landmarks[4].y * image.shape[0])) < 50):
				count += 1
			else:
				count = 0
			if count == 6:
				previous = []
				count = 0
			if previous:
				if distance(int(landmarks[8].x * image.shape[1]),int(landmarks[8].y * image.shape[0]),previous[-1][0],previous[-1][1]) < 100:
					previous.append((int(landmarks[8].x * image.shape[1]), int(landmarks[8].y * image.shape[0])))
			else:
					previous.append((int(landmarks[8].x * image.shape[1]), int(landmarks[8].y * image.shape[0])))

		for (x,y) in landmarks:
			if int(landmarks[8].x * image.shape[1]) > 325 and int(landmarks[8].x * image.shape[1]) < 415 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("C")
			elif int(landmarks[8].x * image.shape[1]) > 415 and int(landmarks[8].x * image.shape[1]) < 505 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("D")
			elif int(landmarks[8].x * image.shape[1]) > 505 and int(landmarks[8].x * image.shape[1]) < 595 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("E")
			elif int(landmarks[8].x * image.shape[1]) > 595 and int(landmarks[8].x * image.shape[1]) < 685 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("F")
			elif int(landmarks[8].x * image.shape[1]) > 685 and int(landmarks[8].x * image.shape[1]) < 775 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("G")
			elif int(landmarks[8].x * image.shape[1]) > 775 and int(landmarks[8].x * image.shape[1]) < 865 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("A")
			elif int(landmarks[8].x * image.shape[1]) > 865 and int(landmarks[8].x * image.shape[1]) < 955 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("B")
		

		if results.right_hand_landmarks: #if image contains landmarks
			landmarks = results.right_hand_landmarks.landmark
			if int(landmarks[8].x * image.shape[1]) > 325 and int(landmarks[8].x * image.shape[1]) < 415 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("C")
			elif int(landmarks[8].x * image.shape[1]) > 415 and int(landmarks[8].x * image.shape[1]) < 505 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("D")
			elif int(landmarks[8].x * image.shape[1]) > 505 and int(landmarks[8].x * image.shape[1]) < 595 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("E")
			elif int(landmarks[8].x * image.shape[1]) > 595 and int(landmarks[8].x * image.shape[1]) < 685 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("F")
			elif int(landmarks[8].x * image.shape[1]) > 685 and int(landmarks[8].x * image.shape[1]) < 775 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("G")
			elif int(landmarks[8].x * image.shape[1]) > 775 and int(landmarks[8].x * image.shape[1]) < 865 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("A")
			elif int(landmarks[8].x * image.shape[1]) > 865 and int(landmarks[8].x * image.shape[1]) < 955 and int(landmarks[8].y * image.shape[0]) < 360:
				playc("B")




			cv2.circle(image,(int(landmarks[8].x * image.shape[1]),int(landmarks[8].y * image.shape[0])),30,(0, 165, 255),-1)
			if(distance(int(landmarks[8].x * image.shape[1]),int(landmarks[8].y * image.shape[0]),int(landmarks[4].x * image.shape[1]),int(landmarks[4].y * image.shape[0])) < 50):
				count += 1
			else:
				count = 0
			if count == 6:
				previous = []
				count = 0
			if previous:
				if distance(int(landmarks[8].x * image.shape[1]),int(landmarks[8].y * image.shape[0]),previous[-1][0],previous[-1][1]) < 100:
					previous.append((int(landmarks[8].x * image.shape[1]), int(landmarks[8].y * image.shape[0])))
			else:
					previous.append((int(landmarks[8].x * image.shape[1]), int(landmarks[8].y * image.shape[0])))
		image[cy:cy + h1, cx:cx + w1] = overlay
		#for i in range(1,len(previous) - 1): 
			#cv2.line(image,previous[i-1],previous[i],(255,0,0),10)
		#if len(previous) > 1:
			#cv2.line(image,previous[-2],previous[-1],(0,0,255),20)
		cv2.imshow("Handtracker", image)
		key = cv2.waitKey(1)
		if key == 27:  # Press 'Esc' key to exit
			break

	cam.release()
	cv2.destroyAllWindows()

def distance(x1,y1,x2,y2):
    return sqrt((y2-y1)**2+(x2-x1)**2)

is_playing = {"A":False,"B":False,"C":False,"D":False,"E":False,"F":False,"G":False,}

def play(s):
    global is_playing
    if not is_playing[s]:
        is_playing[s] = True
        playsound(f"audio/piano2/{s}.wav")
        is_playing[s] = False

def playc(s):
    thread = threading.Thread(target=play, args=(s,))
    thread.start()

test()