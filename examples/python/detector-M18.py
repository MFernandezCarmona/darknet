import pprint

import darknet as dn

net = dn.load_net("/home/manolofc/qi_ws/darknet/cfg/yolov3-tiny.cfg", "/home/manolofc/qi_ws/darknet/yolov3-tiny.weights", 0)
meta = dn.load_meta("/home/manolofc/qi_ws/darknet/cfg/coco.data")



fileList = [
'/home/manolofc/qi_ws/Montreal2018/images/objects/additional/help_me_carry_bag_sspl.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/cleaning_stuff/cloth_sspl.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/cleaning_stuff/scrubby.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/cleaning_stuff/sponge_sspl.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/containers/basket.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/containers/tray.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/drinks/chocolate_drink.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/drinks/coke.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/drinks/grape_juice.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/drinks/orange_juice.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/drinks/sprite.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/food/cereal_sspl.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/food/noodles.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/food/sausages.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/fruits/apple.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/fruits/orange.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/fruits/paprika.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/snacks/crackers.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/snacks/potato_chips.jpg',
'/home/manolofc/qi_ws/Montreal2018/images/objects/snacks/pringles.jpg'
]

detectDict={}

for fil in fileList:
	r = dn.detect(net, meta, fil)
	print r
	objectName = fil.split('/')[-1]
	detectDict[objectName]=len(r)

print detectDict





r = dn.detect(net, meta, "data/person.jpg")
print r

