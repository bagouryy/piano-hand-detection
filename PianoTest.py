#import libraries
from math import sqrt
import os
import cv2
import mediapipe as mp
from playsound import playsound
import threading

def main():
	mp_drawing = mp.solutions.drawing_utils #drawing between dotes
	mp_drawing_styles = mp.solutions.drawing_styles

	mphands = mp.solutions.holistic #track hands in realtime

	# Access webcam
	try:
		cam = cv2.VideoCapture(0)
	except Exception as e:
		print(f"Error: Unable to access webcam. {e}")
		return

	hands = mphands.Holistic(model_complexity=1,
static_image_mode=False,
smooth_landmarks = True,
min_detection_confidence=0.5,
min_tracking_confidence=0.5)
	previous = []
	count = 0

	fingers = [8, 12, 16, 20]

	while True:

		data, image = cam.read()
		if not data:
			print("Error: Unable to read from the webcam.")
			break
		#flip image
		image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
		image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

		image = cv2.resize(image, (1280, 720))

		file_path = 'piano-2.png'
		if os.path.exists(file_path):
			overlay = cv2.imread(file_path)
		else:
			print(f"Error: File '{file_path}' does not exist.")

		overlay = cv2.imread('piano-2.png')
		#n1 = cv2.imread('n-1.png')		
		#n2 = cv2.imread('n-2.png')
		#n3 = cv2.imread('n-3.png')
		#n4 = cv2.imread('n-4.png')


		overlay = cv2.resize(overlay, (int(1280/9 * 7), int(288)))

		#n1 = cv2.resize(n1, (int(1280/9 * 0.8), int(144 * 0.8)))
		#n2 = cv2.resize(n2, (int(1280/9 * 0.8), int(144 * 0.8)))
		#n3 = cv2.resize(n3, (int(1280/9 * 0.8), int(144 * 0.8)))
		#n4 = cv2.resize(n4, (int(1280/9 * 0.8), int(144 * 0.8)))

# Get dimensions of the background image
		h, w = image.shape[:2]
		
		# Get dimensions of the piano overlay image
		h1, w1 = overlay.shape[:2]

		# Get dimensions of n4 overlay
		#hn4, wn4 = n4.shape[:2]
		#hn3, wn3 = n3.shape[:2]
		#hn2, wn2 = n2.shape[:2]
		#hn1, wn1 = n1.shape[:2]


		# Calculate the position to place n4 at the bottom left
		#xn4 = w - hn4 -1
		#yn4 = h - wn4 -10

		#xn3 = w - hn3 -1
		#yn3 = h - wn3 -10 - hn4

		#xn2 = w - hn4 -1
		#yn2 = h - wn4 -10 - hn3 -hn4
		
		#xn1 = w - hn4 -1
		#yn1 = h - wn4 -10 - hn2 - hn3 - hn4

		
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
		#image[yn4:yn4 + hn4, xn4: xn4 + wn4] = n4
		#image[yn3:yn3 + hn3, xn3: xn3 + wn3] = n3
		#image[yn2:yn2 + hn2, xn2: xn2 + wn2] = n2
		#image[yn1:yn1 + hn1, xn1: xn1 + wn1] = n1
		for (x,y) in landmarks:
			if x > 187 and x < 240 and y < 180: # Note for C#
				playc("C1s")
			elif x > 258 and x < 310 and y < 180: # Note for D#
				playc("D1s")
			elif x > 399 and x < 452 and y < 180: # Note for F#
				playc("F1s")
			elif x > 470 and x < 522 and y < 180: # Note for G#
				playc("G1s")
			elif x > 542 and x < 594 and y < 180: # Note for A#
				playc("A1s")
			elif x > 685 and x < 738 and y < 180: # Note for C#
				playc("C2s")
			elif x > 756 and x < 810 and y < 180: # Note for D#
				playc("D2s")
			elif x > 898 and x < 950 and y < 180: # Note for F#
				playc("F2s")
			elif x > 970 and x < 1023 and y < 180: # Note for G#
				playc("G2s")
			elif x > 1040 and x < 1092 and y < 180: # Note for A#
				playc("A2s")
			elif x > 144 and x < 216 and y < 288:
				playc("C1")
			elif x > 216 and x < 288 and y < 288:
				playc("D1")
			elif x > 288 and x < 360 and y < 288:
				playc("E1")
			elif x > 360 and x < 432 and y < 288:
				playc("F1")
			elif x > 432 and x < 504 and y < 288:
				playc("G1")
			elif x > 504 and x < 576 and y < 288:
				playc("A1")
			elif x > 576 and x < 648 and y < 288:
				playc("B1")
			elif x > 648 and x < 720 and y < 288:
				playc("C2")
			elif x > 720 and x < 792 and y < 288:
				playc("D2")
			elif x > 792 and x < 864 and y < 288:
				playc("E2")
			elif x > 864 and x < 936 and y < 288:
				playc("F2")
			elif x > 936 and x < 1008 and y < 288:
				playc("G2")
			elif x > 1008 and x < 1100 and y < 288:
				playc("A2")
			elif x > 1100 and x < 1172 and y < 288:
				playc("B2")
			elif x > 1165 and y > 252 and y < 370: #circle 1:
				fingers= [8]
			elif x > 1165 and y > 370 and y < 480: #circle 2:
				fingers= [8,12]
			elif x > 1165 and y > 480 and y < 605: #circle 4:
				fingers= [8,12,16]
			elif x > 1165 and y > 605: #circle 4:
				fingers= [8,12,16,18]

			cv2.circle(image,(x,y),10,(0, 165, 255),-1)
		cv2.imshow("Handtracker", image)
		key = cv2.waitKey(1)
		if key == 27:  # Press 'Esc' key to exit
			break

	cam.release()
	cv2.destroyAllWindows()

def distance(x1,y1,x2,y2):
    return sqrt((y2-y1)**2+(x2-x1)**2)

is_playing = {"A1":False,"B1":False,"C1":False,"D1":False,"E1":False,"F1":False,"G1":False,"A2":False,"B2":False,"C2":False,"D2":False,"E2":False,"F2":False,"G2":False,"A1s":False,
"C1s":False,"D1s":False,
"F1s":False,"G1s":False,"A2s":False,
"C2s":False,"D2s":False,
"F2s":False,"G2s":False }

def play(s):
    global is_playing
    if not is_playing[s]:
        is_playing[s] = True
        playsound(f"audio/piano3/{s}.wav")
        is_playing[s] = False

def playc(s):
    thread = threading.Thread(target=play, args=(s,))
    thread.start()

main()
