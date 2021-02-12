import sys
import argparse

from utils.tfrecords import create_dataset


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputdir', type=str, default='./images')
    parser.add_argument('-o', '--outputdir', type=str, default='./data')
    parser.add_argument('--train', type=float, default=0.8)
    parser.add_argument('--splits', nargs='+', type=int, default=(1, 1, 1))

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument(
        '--shuffle', dest='random', action='store_true')
    feature_parser.add_argument(
        '--no-shuffle', dest='random', action='store_false')
    parser.set_defaults(random=True)

    args = parser.parse_args()

    create_dataset(
        input_dir=args.inputdir,
        output_dir=args.outputdir,
        training_percentage=args.train,
        random=args.random,
        splits=args.splits
    )
    

    # ? ===== DATASET =====
    # normal: 555
    # ardido: 883
    # brocado: 219
    # marinheiro: 255
    # preto: 459
    # verde: 456

    # ? ===== TREINAMENTO (80%) =====
    # normal: 456
    # ardido: 682
    # brocado: 189
    # marinheiro: 205
    # preto: 359
    # verde: 370
    # TOTAL: 2261

    # ? ===== VALIDAÇÃO (10%) =====
    # normal: 49
    # ardido: 100
    # brocado: 13
    # marinheiro: 27
    # preto: 54
    # verde: 39
    # TOTAL: 282

    # ? ===== TESTE (10%) =====
    # normal: 50
    # ardido: 101
    # brocado: 17
    # marinheiro: 23
    # preto: 46
    # verde: 47
    # TOTAL: 284


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
