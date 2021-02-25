from utils.export_model import export_tolite


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--modeldir', type=str, default='models/CoffeeNet6')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--output', type=int, default='coffeenet6_v1.0.tflite')

    args = parser.parse_args()

    export_tolite(args.modeldir, args.epoch, args.output)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
