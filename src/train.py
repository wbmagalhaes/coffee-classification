import sys
import argparse

from utils.CoffeeNet import load_datasets, create_model, train


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--train', type=str)
    parser.add_argument('-v', '--valid', type=str)
    parser.add_argument('-o', '--output', type=str)

    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--epochs', type=int)

    parser.add_argument('--imsize', type=int, default=64)
    parser.add_argument('--nlayers', type=int, default=5)
    parser.add_argument('--filters', type=int, default=64)
    parser.add_argument('--kernelinit', type=str, default='he_normal')
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--biasinit', type=float, default=0.1)
    parser.add_argument('--lrelualpha', type=float, default=0.02)
    parser.add_argument('--outactivation', type=str, default='softmax')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--labelsmoothing', type=float, default=0.2)

    args = parser.parse_args()

    train_ds, valid_ds, train_steps, valid_steps = load_datasets(
        [args.train],
        [args.valid],
        args.batchsize
    )

    model, cbs = create_model(
        model_name=args.output,
        input_shape=(args.imsize, args.imsize, 3),
        num_layers=args.nlayers,
        filters=args.filters,
        kernel_initializer=args.kernelinit,
        l2=args.l2,
        bias_value=args.biasinit,
        leaky_relu_alpha=args.lrelualpha,
        output_activation=args.outactivation,
        lr=args.lr,
        label_smoothing=args.labelsmoothing
    )

    history = train(
        model,
        train_ds,
        valid_ds,
        train_steps,
        valid_steps,
        epochs=args.epochs,
        callbacks=cbs
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
