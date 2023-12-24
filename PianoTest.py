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

		overlay = cv2.imread('piano-2.png')
		#n1 = cv2.imread('n-1.png')		
		#n2 = cv2.imread('n-2.png')
		#n3 = cv2.imread('n-3.png')
		#n4 = cv2.imread('n-4.png')


		overlay = cv2.resize(overlay, (int(1280/9 * 7), int(288)))

		#n1 = cv2.resize(n1, (int(1280/9), int(144)))
		#n2 = cv2.resize(n2, (int(1280/9), int(144)))
		#n3 = cv2.resize(n3, (int(1280/9), int(144)))
		#n4 = cv2.resize(n4, (int(1280/9), int(144)))

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

		if results.right_hand_landmarks: #if image contains landmarks
			landmarks_right = results.right_hand_landmarks.landmark

		for i in fingers:
				if landmarks_left:
					landmarks.append((int(landmarks_left[i].x * image.shape[1]),int(landmarks_left[i].y * image.shape[0])))
				if landmarks_right:
					landmarks.append((int(landmarks_right[i].x * image.shape[1]),int(landmarks_right[i].y * image.shape[0])))

		image[cy:cy + h1, cx:cx + w1] = overlay
		for (x,y) in landmarks:
			if x > 144 and x < 216 and y < 288:
				playc("C")
			elif x > 187 and x < 240 and y < 180: # Note for C#
				print("C#")
			elif x > 258 and x < 310 and y < 180: # Note for D#
				print("D#")
			elif x > 399 and x < 452 and y < 180: # Note for F#
				print("F#")
			elif x > 470 and x < 522 and y < 180: # Note for G#
				print("G#")
			elif x > 542 and x < 594 and y < 180: # Note for A#
				print("A#")
			elif x > 685 and x < 738 and y < 180: # Note for C#
				print("C#")
			elif x > 756 and x < 810 and y < 180: # Note for D#
				print("D#")
			elif x > 898 and x < 950 and y < 180: # Note for F#
				print("F#")
			elif x > 970 and x < 1023 and y < 180: # Note for G#
				print("G#")
			elif x > 1040 and x < 1092 and y < 180: # Note for A#
				print("A#")
			
			elif x > 216 and x < 288 and y < 288:
				playc("D")
			elif x > 288 and x < 360 and y < 288:
				playc("E")
			elif x > 360 and x < 432 and y < 288:
				playc("F")
			elif x > 432 and x < 504 and y < 288:
				playc("G")
			elif x > 504 and x < 576 and y < 288:
				playc("A")
			elif x > 576 and x < 648 and y < 288:
				playc("B")
			elif x > 648 and x < 720 and y < 288:
				playc("C")
			elif x > 720 and x < 792 and y < 288:
				playc("D")
			elif x > 792 and x < 864 and y < 288:
				playc("E")
			elif x > 864 and x < 936 and y < 288:
				playc("F")
			elif x > 936 and x < 1008 and y < 288:
				playc("G")
			elif x > 1008 and x < 1100 and y < 288:
				playc("A")
			elif x > 1100 and x < 1172 and y < 288:
				playc("B")

			cv2.circle(image,(x,y),10,(0, 165, 255),-1)


			
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