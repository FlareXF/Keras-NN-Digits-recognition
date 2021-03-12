#from pygame import *
import os
import re
import glob
import pygame
from pygame import gfxdraw
import numpy as np
import scipy as sp
import scipy.ndimage

import keras
from keras.models import load_model

def make_predict(model,image):
	image = image/255
	img_arr = image.reshape(1,28,28,1)
	print(model.predict_classes([img_arr]))
	return model.predict_classes([img_arr])



models = os.listdir('models')


W,D=640,672
D_paint = D-34
L_W,L_D=28,28
freq = 2500
state = 1
x,y=0,0
btn_prs = 0
new_x,new_y=0,0
sigma = 2
type_of_inter = 1
answ = -1
counter = 0
model_set = 1
arrow = 0
take_scr = 0
gr = 140
text_models = np.empty((len(models),2)).tolist()
text = np.empty((10,2)).tolist()


WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (140,140,140)
LOW_GRAY = (180,180,180)
DARK = (80,80,80)

pygame.init()
screen = pygame.display.set_mode((W,D))
low = pygame.Surface((L_W,L_D))
gauss_surf = low
pygame.display.set_caption('')
clock = pygame.time.Clock()
screen.fill(BLACK)
low.fill(0)
screen.blit(low,(50,25))

font = pygame.font.Font('spectrum.ttf',32)
font_1 = pygame.font.Font('spectrum.ttf',18)


pygame.draw.rect(screen,GRAY,(0,D_paint,640,50))

def text_show():
	global text
	global text_models
	global arrow
	global take_scr
	global gr
	maxi = [0]
	clor = LOW_GRAY
	pygame.draw.rect(screen,GRAY,(0,D_paint,640,50))
	if model_set == 1:
		shift_x = 18
		len_models = len(models)
		arrow = 0 if arrow <= 0 else arrow
		arrow = len_models-1 if arrow >= len_models else arrow
		for i in range(len_models):
			maxi.append(len(models[i]))
			pygame.draw.rect(screen,GRAY,(0,shift_x,max(maxi)*15.4,len_models*20))
			text_models[i][0] = (font_1.render(models[i], False, WHITE))
			text_models[i][1] = (0,shift_x+(i*20))
			pygame.draw.rect(screen,LOW_GRAY,(0,shift_x+(arrow*20),len(models[arrow])*15.4,18))
		screen.blits(text_models)
	for i in range(10):
		if i == answ:
			clor = BLACK
			pygame.draw.rect(screen,WHITE,(i*67.5,D_paint,32,35))
		if answ == 10:
			clor = BLACK
			for z in range(10):
				pygame.draw.rect(screen,WHITE,(i*67.5,D_paint,32,35))
		text[i][0] = (font.render(str(i), False, clor))
		text[i][1] = (i*67.5,640)
		clor = LOW_GRAY

	text_1 = font_1.render(str(input_n[new_x,new_y]) if new_y < 28 else '[0 0 0]', False, GRAY)
	text_2 = font_1.render(str(sigma), False, GRAY)
	text_3 = font_1.render('Gauss' if type_of_inter == 0 else 'uniform', False, GRAY)
	screen.blit(text_1,(0,0))
	screen.blit(text_2,(200,0))
	screen.blit(text_3, (270,0))
	screen.blits(text)
	if take_scr >= 0.01:
		tex = 'Screenshot saved!'
		if take_scr <= 0.05 and gr > 0:
			gr -= 1
		text_scr = font_1.render(tex, False, (gr,gr,gr))
		screen.blit(text_scr,(0,D_paint-18))
		take_scr /= 1.006

def gray(im):
    im = (255*im)/16777215
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret
def take_image():
	global counter
	global take_scr
	global gr
	temp=[]
	take_scr,gr = 1,140
	screen_list = os.listdir('h_dataset')
	for i in range(len(screen_list)):
		temp.append(re.findall(r'\d+',screen_list[i]))
	if np.shape(temp) != (0,):
		counter = max(list(map(int,np.reshape(temp,len(screen_list)))))
	counter += 1
	pygame.image.save(gauss_surf, 'h_dataset\\sign_'+str(counter)+'.png')
while state:
	clock.tick(freq)

	x,y = pygame.mouse.get_pos()
	new_x = (L_W*x)//W
	new_y = (L_D*y)//(D_paint)


	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			state = 0
		if event.type == pygame.MOUSEBUTTONDOWN:
			if model_set == 0:
				if event.button == 1:
					btn_prs = 1
					color = WHITE
				if event.button == 2:
					btn_prs = 1
					color = BLACK
				if event.button == 3:
					low.fill(BLACK)
					answ = -1

		if model_set == 0:
			if event.type == pygame.MOUSEBUTTONUP:
				if input_n[:,:,0].any() == 1:
					answ = make_predict(model,np.transpose(input_n[:,:,0]))
				btn_prs = 0

		if event.type == pygame.KEYDOWN:
			if model_set == 0:
				sigma = (sigma-coef) if event.key == pygame.K_COMMA else sigma
				sigma = (sigma+coef) if event.key == pygame.K_PERIOD else sigma

				type_of_inter = 1 if event.key == pygame.K_F1 else type_of_inter
				type_of_inter = 0 if event.key == pygame.K_F2 else type_of_inter

				take_image() if event.key == pygame.K_s else True

			arrow = arrow-1 if event.key == pygame.K_UP else arrow
			arrow = arrow+1 if event.key == pygame.K_DOWN else arrow
			model_set = 1 if event.key == pygame.K_F12 else model_set
			if event.key == pygame.K_RETURN:
				model_set = 0
				model = load_model('models\\'+models[arrow])
				#print (model.summary())


	if btn_prs == 1:
		pygame.draw.circle(low,color,(new_x,new_y),1)
		#pygame.gfxdraw.filled_circle(low,new_x,new_y,1,color)


	sigma = 0 if sigma <= 0 else sigma
	arrow = 0 if arrow <= 0 else arrow
	coef = 0.1 if type_of_inter == 0 else 1
	rd_C = 1 if type_of_inter == 0 else 0
	sigma = round(sigma,rd_C)








	input_n = np.copy(pygame.PixelArray(low))
	input_n = sp.ndimage.filters.gaussian_filter(input_n,sigma=sigma) if type_of_inter == 0 else input_n
	input_n = sp.ndimage.filters.uniform_filter(input_n,sigma) if type_of_inter == 1 else input_n

	input_n = gray(input_n)
	gauss_surf = pygame.surfarray.make_surface(input_n)
	screen.blit(pygame.transform.scale(gauss_surf,(W,D_paint)),(0,0))
	text_show()
	pygame.display.flip()