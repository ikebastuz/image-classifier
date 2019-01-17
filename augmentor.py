import Augmentor
import shutil

dataPath = "trainset/1"

p_flip = Augmentor.Pipeline(dataPath)
p_flip.flip_left_right(probability=1)
p_flip.process()

p_zoom = Augmentor.Pipeline(dataPath)
p_zoom.zoom(probability=1, min_factor=1.05, max_factor=1.2)
p_zoom.process()

p_zoom_flip = Augmentor.Pipeline(dataPath)
p_zoom_flip.flip_left_right(probability=1)
p_zoom_flip.zoom(probability=1, min_factor=1.05, max_factor=1.2)
p_zoom_flip.process()

shutil.move(dataPath + '/output', dataPath + '_aug')
'''
# Testset
test_pipe = Augmentor.Pipeline(dataPath)

test_pipe.flip_left_right(probability=0.5)
test_pipe.zoom(probability=0.5, min_factor=1.05, max_factor=1.2)
test_pipe.random_distortion(
    probability=1, grid_width=4, grid_height=4, magnitude=8)
test_pipe.sample(200)

shutil.move(dataPath + '/output', dataPath + '_test')
'''
