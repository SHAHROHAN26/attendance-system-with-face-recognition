'''# facerec.py
import cv2, sys, numpy, os
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
# Part 1: Create fisherRecognizer
print('Recognizing Face Please Be in sufficient Light Conditions...')
# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
	for subdir in dirs:
		names[id] = subdir
		subjectpath = os.path.join(datasets, subdir)
		for filename in os.listdir(subjectpath):
			path = subjectpath + '/' + filename
			lable = id
			images.append(cv2.imread(path, 0))
			lables.append(int(lable))
		id += 1
(width, height) = (130, 100)

# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)

# Part 2: Use fisherRecognizer on camera stream
#print(dir(cv2.face))
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
while True:
	(_, im) = webcam.read()
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
		face = gray[y:y + h, x:x + w]
		face_resize = cv2.resize(face, (width, height))
		# Try to recognize the face
		prediction = model.predict(face_resize)
		cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

		if prediction[1]<500:
			cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
		else:
			cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

	cv2.imshow('OpenCV', im)
    
	key = cv2.waitKey(10)
	if key == 27:
		break


import face_recognition
import numpy as np
import cv2, queue, threading, time
import requests, os, re
import xlsxwriter

class VideoCapture:
	def __init__(self,name):
		self.cap = cv2.VideoCapture(name)
		self.q = queue.Queue()
		t = threading.Thread(target=self._reader)
		t.daemon = True
		t.start()
	def _reader(self):
		while True:
			ret, frame = self.cap.read()
			if not ret:
				break
			if not self.q.empty():
				try:
					self.q.get_nowait()
				except queue.Empty:
					pass
			self.q.put(frame)
	def read(self):
		return self.q.get()
video_capture = VideoCapture(0)
know_face_encodings = []
know_face_names = []
know_faces_filenames = []

for(dirpath, dirnames, filenames) in os.walk('assets/img/users/'):
	know_faces_filenames.extend(filenames)
	break
for filename in know_faces_filenames:
	face=face_recognition.load_image_file('assets/img/users/'+filename)
	know_face_names.append(re.sub("[0-9]",'',filename[:-4]))
	know_face_encodings.append(face_recognition.face_encodings(face)[0])
	
	
face_locations=[]
face_encodings=[]
face_names=[]
process_this_frame=True

while True:
	frame = video_capture.read()
	if process_this_frame:
		face_locations=face_recognition.face_locations(frame)
		face_encodings=face_recognition.face_encodings(frame,face_locations)
		face_names=[]
		
		for face_encoding in face_encodings:
			matches=face_recognition.compare_faces(know_face_encodings, face_encoding)
			name="Unknown"
			face_distances=face_recognition.face_distance(know_face_encodings,face_encoding)
			best_match_index=np.argmin(face_distances)
			if matches[best_match_index]:
				name=know_face_names[best_match_index]
				print("Status",name)
			
			face_names.append(name)
			
	process_this_frame= not process_this_frame
	for(top,right,bottom,left),name in zip(face_locations, face_names):
		cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
		font=cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame,name,(left+6,bottom-6),font,1.0,(255,255,255),1)
	
	cv2.imshow('Video',frame)
	
	#workbook= xlsxwriter.Workbook('abc.xlsx')
	#worksheet=workbook.add_worksheet()
	
	#row=0
	#column=0
	#content=[face_names]
	#for item in content:
	#	worksheet.write(row,column,item)
	#	row +=1
	#workbook.close()
	
	
	if cv2.waitKey(1) & 0xFF ==ord('q'):
		break
video_capture.release()
cv2.destroyAllWindows()
'''

import face_recognition
import numpy as np
import cv2, queue, threading, time
import requests, os, re
import xlsxwriter
import csv

class VideoCapture:
	def __init__(self,name):
		self.cap = cv2.VideoCapture(name)
		self.q = queue.Queue()
		t = threading.Thread(target=self._reader)
		t.daemon = True
		t.start()
	def _reader(self):
		while True:
			ret, frame = self.cap.read()
			if not ret:
				break
			if not self.q.empty():
				try:
					self.q.get_nowait()
				except queue.Empty:
					pass
			self.q.put(frame)
	def read(self):
		return self.q.get()
video_capture = VideoCapture(0)
know_face_encodings = []
know_face_names = []
know_faces_filenames = []
with open('data.csv','r') as f:
		data = csv.reader(f)
		lines = list(data)
for line in lines[1:]:
	line[-1]='absent'
def  markPresent(name):
	with open('data.csv','r') as f:
		data = csv.reader(f)
		lines = list(data)
# for line in lines:
#     line.pop(0)
# print(lines)
		
		for line in lines:
			if line[1] == name:
				line[-1] = 'present'
				with open('data.csv','w') as g:
					writer = csv.writer(g,lineterminator='\n')
					writer.writerows(lines)
					break
def getdata():
	with open('data.csv','r') as f:
		data = csv.reader(f)
		next(data)
		lines = list(data)
		for line in lines:
			names[int(line[0])] = line[1]
			
def update_Excel():
        with open('data.csv') as f:
            data = csv.reader(f)
            lines = list(data)
            for line in lines:
                line.pop(0)
            with open('data.csv','w') as g:
                writer = csv.writer(g,lineterminator='\n')
                writer.writerows(lines)
                
        df = pd.read_csv('data.csv')
        df.to_excel('data.xlsx',index = False)

def from_excel_to_csv():
	df = pd.read_excel(filename,index=False)
	df.to_csv('./data.csv')



	
for(dirpath, dirnames, filenames) in os.walk('assets/img/users/'):
	know_faces_filenames.extend(filenames)
	break
for filename in know_faces_filenames:
	face=face_recognition.load_image_file('assets/img/users/'+filename)
	know_face_names.append(re.sub("[0-9]",'',filename[:-4]))
	know_face_encodings.append(face_recognition.face_encodings(face)[0])
	
	
face_locations=[]
face_encodings=[]
face_names=[]
process_this_frame=True

while True:
	frame = video_capture.read()
	if process_this_frame:
		face_locations=face_recognition.face_locations(frame)
		face_encodings=face_recognition.face_encodings(frame,face_locations)
		face_names=[]
		
		for face_encoding in face_encodings:
			matches=face_recognition.compare_faces(know_face_encodings, face_encoding)
			name="Unknown"
			face_distances=face_recognition.face_distance(know_face_encodings,face_encoding)
			best_match_index=np.argmin(face_distances)
			if matches[best_match_index]:
				name=know_face_names[best_match_index]
				print("Status",name)
				markPresent(name)
					
			
			face_names.append(name)
			
	process_this_frame= not process_this_frame
	for(top,right,bottom,left),name in zip(face_locations, face_names):
		cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
		font=cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame,name,(left+6,bottom-6),font,1.0,(255,255,255),1)
	
	cv2.imshow('Video',frame)
	
	#workbook= xlsxwriter.Workbook('abc.xlsx')
	#worksheet=workbook.add_worksheet()
	
	#row=0
	#column=0
	#content=[face_names]
	#for item in content:
	#	worksheet.write(row,column,item)
	#	row +=1
	#workbook.close()
	
	
	if cv2.waitKey(1) & 0xFF ==ord('q'):
		break
video_capture.release()
cv2.destroyAllWindows()
'''

import face_recognition
import numpy as np
import cv2, queue, threading, time
import requests, os, re
import xlsxwriter
import csv
import datetime

#date=datetime.datetime.now()

class VideoCapture:
	def __init__(self,name):
		self.cap = cv2.VideoCapture(name)
		self.q = queue.Queue()
		t = threading.Thread(target=self._reader)
		t.daemon = True
		t.start()
	def _reader(self):
		while True:
			ret, frame = self.cap.read()
			if not ret:
				break
			if not self.q.empty():
				try:
					self.q.get_nowait()
				except queue.Empty:
					pass
			self.q.put(frame)
	def read(self):
		return self.q.get()
video_capture = VideoCapture(0)
know_face_encodings = []
know_face_names = []
know_faces_filenames = []

def  markPresent(name):
	filename = datetime.datetime.now().strftime('attendance-%Y-%m-%d-%H-%M.csv')
	with open(filename, 'w+') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',')
		#filewriter.writerow(['id', 'name' , 'Enrollment', 'Attendance'])
		filewriter.writerow(['1', 'rohan', '2019' ,'absent'])
		filewriter.writerow(['2', 'aditya', '2001', 'absent'])
		filewriter.writerow(['3', 'dhrites', '2006', 'absent'])
		filewriter.writerow(['4', 'poojan', '2018', 'absent'])

	
	with open(filename,'r') as f:
		data = csv.reader(f)
		lines = list(data)
#for line in lines:
#	 line.pop(0)
# print(lines)
		for line in lines:
			if line[1] == name:
				line[-1] = 'present'
				with open(filename,'w') as g:
					writer = csv.writer(g,lineterminator='\n')
					writer.writerows(lines)
					break
def getdata():
	with open('data.csv','r') as f:
		data = csv.reader(f)
		next(data)
		lines = list(data)
		for line in lines:
			names[int(line[0])] = line[1]
			
def update_Excel():
        with open('data.csv') as f:
            data = csv.reader(f)
            lines = list(data)
            for line in lines:
                line.pop(0)
            with open('data.csv','w') as g:
                writer = csv.writer(g,lineterminator='\n')
                writer.writerows(lines)
                
        df = pd.read_csv('data.csv')
        df.to_excel('data.xlsx',index = False)

def from_excel_to_csv():
	df = pd.read_excel(filename,index=False)
	df.to_csv('./data.csv')



	
for(dirpath, dirnames, filenames) in os.walk('assets/img/users/'):
	know_faces_filenames.extend(filenames)
	break
for filename in know_faces_filenames:
	face=face_recognition.load_image_file('assets/img/users/'+filename)
	know_face_names.append(re.sub("[0-9]",'',filename[:-4]))
	know_face_encodings.append(face_recognition.face_encodings(face)[0])
	
	
face_locations=[]
face_encodings=[]
face_names=[]
process_this_frame=True

while True:
	frame = video_capture.read()
	if process_this_frame:
		face_locations=face_recognition.face_locations(frame)
		face_encodings=face_recognition.face_encodings(frame,face_locations)
		face_names=[]
		
		for face_encoding in face_encodings:
			matches=face_recognition.compare_faces(know_face_encodings, face_encoding)
			name="Unknown"
			face_distances=face_recognition.face_distance(know_face_encodings,face_encoding)
			best_match_index=np.argmin(face_distances)
			if matches[best_match_index]:
				name=know_face_names[best_match_index]
				print("Status",name)
				markPresent(name)
					
			
			face_names.append(name)
			
	process_this_frame= not process_this_frame
	for(top,right,bottom,left),name in zip(face_locations, face_names):
		cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
		font=cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame,name,(left+6,bottom-6),font,1.0,(255,255,255),1)
	
	cv2.imshow('Video',frame)
	
	#workbook= xlsxwriter.Workbook('abc.xlsx')
	#worksheet=workbook.add_worksheet()
	
	#row=0
	#column=0
	#content=[face_names]
	#for item in content:
	#	worksheet.write(row,column,item)
	#	row +=1
	#workbook.close()
	
	
	if cv2.waitKey(1) & 0xFF ==ord('q'):
		break
video_capture.release()
cv2.destroyAllWindows()
'''		