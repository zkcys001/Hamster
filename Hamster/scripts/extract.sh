#python extract.py -m resnet18 && python extract.py -m resnet34 && python extract.py -m resnet50 && python extract.py -m resnet101 && python extract.py -m resnet152
#python extract.py -m resnet18 --method mealv2 && python extract.py -m resnet50 --method mealv2
#python extract.py -m vit_t && python extract.py -m vit_s && python extract.py -m vit_b && python extract.py -m vit_l
# && python extract.py -m deit_t && python extract.py -m deit_s && python extract.py -m deit_b
# && python extract.py -m swim_t && python extract.py -m swim_s && python extract.py -m swim_b && python extract.py -m swim_l
# python extract.py -m vgg11_bn && python extract.py -m vgg13_bn && python extract.py -m vgg16_bn && python extract.py -m vgg19_bn
python extract.py -m cifar_resnet18 && python extract.py -m cifar_resnet34 && python extract.py -m cifar_resnet50

