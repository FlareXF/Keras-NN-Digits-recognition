#from pygame import *
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
	return model.predict_classes([img_arr])


#model = keras.models.load_model('cnn_digs.h5')
model_name = str(input('Model name ->'))
model = load_model('models\\'+model_name+'.h5')


W,D=640,672
L_W,L_D=28,28
freq = 2500
state = 1
x,y=0,0
btn_prs = 0
new_x,new_y=0,0
sigma = 2
type_of_inter = 1
answ = -1


WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (140,140,140)
LOW_GRAY = (180,180,180)

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


pygame.draw.rect(screen,GRAY,(0,638,640,50))

def text_show():
	global text
	clor = LOW_GRAY
	pygame.draw.rect(screen,GRAY,(0,638,640,50))
	for i in range(10):
		if i == answ:
			clor = BLACK
			pygame.draw.rect(screen,WHITE,(i*67.5,639,32,35))
		text = font.render(str(i), False, clor)
		screen.blit(text,(i*67.5,640))
		clor = LOW_GRAY
	text_1 = font_1.render(str(input_n[new_x,new_y]) if new_y < 28 else '[0 0 0]', False, GRAY)
	text_2 = font_1.render(str(sigma), False, GRAY)
	text_3 = font_1.render('Gauss' if type_of_inter == 0 else 'uniform', False, GRAY)
	screen.blit(text_1,(0,0))
	screen.blit(text_2,(200,0))
	screen.blit(text_3, (270,0))

def gray(im):
    im = (255*im)/16777215
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret

while state:
	clock.tick(freq)

	x,y = pygame.mouse.get_pos()
	new_x = (L_W*x)//W
	new_y = (L_D*y)//(D-34)


	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			state = 0
		if event.type == pygame.MOUSEBUTTONDOWN:
			if event.button == 1:
				btn_prs = 1
				color = WHITE
			if event.button == 2:
				btn_prs = 1
				color = BLACK
			if event.button == 3:
				low.fill(BLACK)
				answ = -1

		if event.type == pygame.MOUSEBUTTONUP:
			if input_n[:,:,0].any() == 1:
				answ = make_predict(model,np.transpose(input_n[:,:,0]))
			btn_prs = 0

		if event.type == pygame.KEYDOWN:

			sigma = (sigma-coef) if event.key == pygame.K_COMMA else sigma
			sigma = (sigma+coef) if event.key == pygame.K_PERIOD else sigma

			type_of_inter = 1 if event.key == pygame.K_F1 else type_of_inter
			type_of_inter = 0 if event.key == pygame.K_F2 else type_of_inter


	if btn_prs == 1:
		#x,y = pygame.mouse.get_pos()
		#new_x = (L_W*x)//W
		#new_y = (L_D*y)//(D-34)

		pygame.draw.circle(low,color,(new_x,new_y),1)
		#pygame.gfxdraw.filled_circle(low,new_x,new_y,1,color)

	pressed_keys = pygame.key.get_pressed()
	sigma = 0 if sigma <= 0 else sigma
	coef = 0.1 if type_of_inter == 0 else 1
	rd_C = 1 if type_of_inter == 0 else 0
	sigma = round(sigma,rd_C)








	input_n = np.copy(pygame.PixelArray(low))
	input_n = sp.ndimage.filters.gaussian_filter(input_n,sigma=sigma) if type_of_inter == 0 else input_n
	input_n = sp.ndimage.filters.uniform_filter(input_n,sigma) if type_of_inter == 1 else input_n
	input_n = gray(input_n)
	gauss_surf = pygame.surfarray.make_surface(input_n)
	screen.blit(pygame.transform.scale(gauss_surf,(640,638)),(0,0))
	text_show()
	pygame.display.flip()