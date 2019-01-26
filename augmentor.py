import Augmentor
import shutil

dataPath = "trainset/sum_1_2"

p = Augmentor.Pipeline(dataPath)

p.flip_left_right(probability=1)
p.process()

p.flip_top_bottom(probability=1)
p.process()

p.rotate90(probability=1)
p.process()

p.rotate270(probability=1)
p.process()

p.zoom(probability=1, min_factor=1.05, max_factor=1.2)
p.process()


shutil.move(dataPath + '/output', dataPath + '_aug360')
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
