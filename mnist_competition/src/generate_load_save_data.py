from mnist import MNIST
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from os import listdir
import cv2 as cv
import time
import sys

im_out_size=64 #Taille de l'image en sortie
rot_max=15 #Nombre de degrés de rotation maximal

def make_image(im_data,vect_only=False):
	if vect_only:
		imsize=int(math.sqrt(len(im_data)))
		im=np.array(im_data,dtype=np.uint8)
	else:
		imsize=int(math.sqrt(len(im_data[0])))
		im=np.array(im_data[0],dtype=np.uint8)
	im=np.reshape(im,(imsize,imsize))
	return im

def generate_output_im(Indices,rotate=False,background=False,custom_background=False):
	elements=[data_digits[Indices[0]],data_digits[Indices[1]],data_am[Indices[2]]]
	images_array=[make_image(elements[i]) for i in range(len(elements))]
	labels_current=[elements[i][1] for i in range(len(elements))]
	if labels_current[2]=="a":
		result=labels_current[0]+labels_current[1]
		op="+"
	else:
		result=labels_current[0]*labels_current[1]
		op="x"
	title="%d %s %d = %d" %(labels_current[0],op,labels_current[1],result)
	output_data={"result":result,"digit1":labels_current[0],"digit2":labels_current[1],"title":title,"op":op}
	while(1):
		nb_l=0
		im=np.zeros((im_out_size,im_out_size),dtype=np.uint8)
		for i,l in enumerate(images_array):
			key="bbox %d" %i
			if rotate:
				l=ndimage.rotate(l,random.randrange(-rot_max,rot_max))
			lsize=l.shape[0]
			xymax=im_out_size-lsize
			if i==0:
				xy=[random.randrange(0, xymax) for i in range(2)]
				im[xy[0]:xy[0]+lsize,xy[1]:xy[1]+lsize]+=l
				output_data[key]=xy
				nb_l+=1
			else:
				found=False
				k=0
				while(not found and k<10):
					im_temp=np.zeros((im_out_size,im_out_size),dtype=np.uint8)
					xy=[random.randrange(0, xymax) for i in range(2)]
					im_temp[xy[0]:xy[0]+lsize,xy[1]:xy[1]+lsize]+=l
					if sum(sum(im_temp*im))==0:
						im=im+im_temp
						output_data[key]=xy
						nb_l+=1
						found=True
					else:
						k+=1
		if nb_l==len(elements):
			if background:
				if custom_background:
					im_b=np.array(images_b[Indices[3]])
				else:
					im_b=np.array(images_b[random.randrange(0, len(images_b))])
				im_out = cv.addWeighted(im, 0.5, im_b, 0.5,0) 
			else:
				im_out=im
			output_data["im"]=make_vect(im_out)
			return output_data


def disp_im(im_data):
	im=make_image(im_data["im"],vect_only=True)
	plt.imshow(im,cmap=plt.get_cmap('gray'))
	plt.title(im_data["title"])
	plt.show()

def make_vect(im_array):
	im_vect=[item for sublist in im_array for item in sublist]
	return im_vect

def random_instance():
	return generate_output_im([random.randrange(0, len(data_digits)),random.randrange(0, len(data_digits)),random.randrange(0, len(data_am))],rotate=True,background=True)

def generate_backgrounds():
	path="./Background patterns/"
	images_b=[]
	images_path=[path+imp for imp in listdir(path)]
	for im_path in images_path:
		im = cv.cvtColor(cv.imread(im_path), cv.COLOR_RGB2GRAY)
		xn=int(math.floor(im.shape[0]/im_out_size))
		yn=int(math.floor(im.shape[1]/im_out_size))
		for i in range(0,xn):
			for j in range(0,yn):
				images_b.append(im[i*im_out_size:(i+1)*im_out_size,j*im_out_size:(j+1)*im_out_size])
	return images_b

def disp_digit(im_data,title="title",vect=True):
	if vect:
		im=make_image(im_data,vect_only=True)
	plt.imshow(im,cmap=plt.get_cmap('gray'))
	plt.title(title)
	plt.show()

def import_am():
	mndata_l=MNIST('./EMNIST/')
	mndata_l.select_emnist('letters')
	images_l,labels_l=mndata_l.load_training()
	a_ind= [i for i, x in enumerate(labels_l) if x == 1]
	m_ind= [i for i, x in enumerate(labels_l) if x == 13]
	images_am=[images_l[i] for i in range(len(images_l)) if i in a_ind]+[images_l[i] for i in range(len(images_l)) if i in m_ind]
	labels_am=["a" for i in range(len(a_ind))]+["m" for i in range(len(m_ind))]
	data_am=[[images_am[i],labels_am[i]] for i in range(len(images_am))]
	return data_am

def import_digits():
	mndata=MNIST('./samples/')
	images, labels = mndata.load_training()
	data_digits=[[images[i],labels[i]] for i in range(len(images))]
	return data_digits


def save_data(instances,save_path="./"):
	im_data=[instances[i]['im'] for i in range(len(instances))]
	digit1_data=[instances[i]['digit1'] for i in range(len(instances))]
	digit2_data=[instances[i]['digit2'] for i in range(len(instances))]
	op_data=[instances[i]['op'] for i in range(len(instances))]
	np.savez(save_path+"x_train.npz",im_data)
	np.savez(save_path+"y_train.npz",digit1_data,digit2_data,op_data)


def load_data(save_path="./"):
	x_data=np.load(save_path+"x_train.npz")
	y_data=np.load(save_path+"y_train.npz")
	im_data=x_data['arr_0']
	digit1_data=y_data['arr_0']
	digit2_data=y_data['arr_1']
	op_data=y_data['arr_2']
	instances=[]
	for i in range(len(im_data)):

		inst={'im':im_data[i],'digit1':digit1_data[i],'digit2':digit2_data[i],'op':op_data[i]}
		if inst['op']=="+":
			inst["result"]=inst["digit1"]+inst["digit2"]
		else:
			inst["result"]=inst["digit1"]*inst["digit2"]
		
		inst['title']="%d %s %d = %d" %(inst["digit1"],inst["op"],inst["digit2"],inst['result'])
		instances.append(inst)
	return instances

def generate_instances(nbr_inst):
	nbr_current=0
	ind_digit1=([ind for ind in range(len(data_digits)) for i in range(int(math.ceil(nbr_inst/len(data_digits))))])
	ind_digit2=([ind for ind in range(len(data_digits)) for i in range(int(math.ceil(nbr_inst/len(data_digits))))])
	random.shuffle(ind_digit1)
	random.shuffle(ind_digit2)
	ind_digit1=ind_digit1[0:nbr_inst]
	ind_digit2=ind_digit2[0:nbr_inst]

	ind_am=([ind for ind in range(len(data_am)) for i in range(int(math.ceil(nbr_inst/len(data_am))))])
	random.shuffle(ind_am)
	ind_am=ind_am[0:nbr_inst]

	ind_b=([ind for ind in range(len(images_b)) for i in range(int(math.ceil(nbr_inst/(len(images_b)))))])
	random.shuffle(ind_b)
	ind_b=ind_b[0:nbr_inst]

	instances=[]
	completion=0
	next_i=0
	t1=time.time()

	for i in range(nbr_inst):
		if i>=next_i:
			if i>0:
				tc=time.time()-t1
				t_tot=100*tc/completion
				t_restant=t_tot-tc
				t_string=str(t_restant)+" s"
			else:
				t_string="Unknown"
			out_string="Progress = %d%% (%d/%d). Estimated time = %s \n" %(completion,i+1,nbr_inst,t_string)
			sys.stdout.write(out_string)
			sys.stdout.flush()
			completion+=5
			next_i=int(completion/100*(nbr_inst-1))
		indices=[ind_digit1[i],ind_digit2[i],ind_am[i],ind_b[i]]
		instances.append(generate_output_im(indices,rotate=True,background=True,custom_background=True))
	return instances


##### Load instances et show exemples ####

# start=time.time()
# instances=load_data()
# end=time.time()
# out_str="Time to load= %f s\n" %(end-start)
# sys.stdout.write(out_str)
# sys.stdout.flush()
# selection=int(input("Number of samples to display: "))
# for i in range(selection):
# 	disp_im(instances[random.randrange(0,len(instances))])


##### Génère et sauvegarde exemplaires ####

# images_b=generate_backgrounds()
# data_am=import_am()
# data_digits=import_digits()
# sys.stdout.write("Loading Completed!\n")
# sys.stdout.flush()
# nbr_inst=50000
# start=time.time()
# instances=generate_instances(nbr_inst)
# end=time.time()
# time_current=end-start
# time_estimated=50000*time_current/nbr_inst

# start=time.time()
# save_data(instances)
# end=time.time()

# print()
# print("Time to generate %d instances: %f s\n" %(nbr_inst,time_current))
# print("Estimated time to generate 50 000: %f s\n" %(time_estimated))
# print("time to save data= %f s" %(end-start))

