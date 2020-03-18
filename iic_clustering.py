import argparse
from IIC import IIC_clustering


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="IIC clustering network"
    )
    parser.add_argument("--path",type=str, required=True, help="Define path to image data")
    parser.add_argument("--input_shape", nargs="+",
                        help="Specifiy shape of image data h,w,c", required=True)
    parser.add_argument("--heads", type=int, default=2,
                        help="Specify number of output softmax layer heads")
    parser.add_argument("--batch_size", type=int,
                        default=256, help="Define batch size")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Epochs to be trained")
    parser.add_argument("--clusters", type=int, required=True,
                        help="Number of output cluster")
    parser.add_argument("--CNN_base", type=str, default="vgg",
                        help="Defines conv base network. Possibilities are: \n -ResNet size > (200,200)\n -Vgg, \n -Mini: MobileNet size > (200,200)\n. ResNet and Mini are pretrained networks")
    parser.add_argument("--crop_images", type=int,
                        default=4, help="size of image cropping")
    parser.add_argument("--aux_cluster", type=bool, default=True,
                        help="Enable auxiliary clustering for noise reduction")
    options = vars(parser.parse_args())
    shape = [int(i) for i in options['input_shape']]

    model = IIC_clustering(path=options['path'], heads = int(options['heads']),batch_size=int(options['batch_size']),
                epochs= int(options['epochs']), z_dimension=int(options['clusters']), crop_image=int(options['crop_images']),
                input_shape=shape, CNN_base=options['CNN_base'],aux_cluster=bool(options['aux_cluster']))
    model.fit()


#python iic_clustering.py --path /home/michael/Downloads/mnist/mnist_png/training --input_shape 28 28 3 --epochs 2 --clusters 10 --aux_cluster 1 
