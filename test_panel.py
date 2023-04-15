from roadSeekerIo import utils
from roadSeekerIo.panel import Panel

panel = Panel("resources/train_detection/00007.png", (200, 200), (100, 100), 1.34234243)
utils.save_panels("resources/train_detection/00007.png", [panel])