import pygame
import time
# pygame.init()
# pygame.mixer.init()
pygame.display.set_mode()
screen_size = (400, 400)
screen = pygame.display.set_mode(screen_size)
screen.fill((0,192,0))
while True:
	pass
	# for event in pygame.event.get():
	# 	print(event)
	# 	if event.type == pygame.QUIT:
	# 		pygame.quit(); #sys.exit() if sys is imported
	# 	if event.type == pygame.KEYDOWN:
	# 		if event.key == pygame.K_0:
	# 			print("Hey, you pressed the key, '0'!")
	# 		if event.key == pygame.K_1:
	# 			print("Doing whatever")
	pygame.event.pump()
	keys = pygame.key.get_pressed()
	print keys[pygame.K_LEFT]
	
	print "Sleeping"
	time.sleep(1)
	#pygame.time.delay(1000)
	print "Sleeping complete"

