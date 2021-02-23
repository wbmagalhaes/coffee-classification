import sys
import argparse

from utils.tfrecords import create_dataset


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputdir', type=str, default='./images')
    parser.add_argument('-o', '--outputdir', type=str, default='./data')
    parser.add_argument('--train_percent', type=float, default=0.8)
    parser.add_argument('--n_files', nargs='+', type=int, default=(1, 1, 1))

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--shuffle', dest='random', action='store_true')
    feature_parser.add_argument('--no-shuffle', dest='random', action='store_false')
    parser.set_defaults(random=True)

    args = parser.parse_args()

    create_dataset(
        input_dir=args.inputdir,
        output_dir=args.outputdir,
        training_percentage=args.train,
        random=args.random,
        n_files=args.n_files
    )

    # ? ===== DATASET =====
    # normal: 1149
    # ardido: 1139
    # brocado: 404
    # marinheiro: 307
    # preto: 615
    # verde: 661
    # TOTAL: 4275

    # ? ===== TREINAMENTO (80%) =====
    # normal: 912
    # ardido: 910
    # brocado: 334
    # marinheiro: 241
    # preto: 500
    # verde: 523
    # TOTAL: 3420

    # ? ===== VALIDAÇÃO (10%) =====
    # normal: 114
    # ardido: 114
    # brocado: 39
    # marinheiro: 31
    # preto: 53
    # verde: 76
    # TOTAL: 427

    # ? ===== TESTE (10%) =====
    # normal: 123
    # ardido: 115
    # brocado: 31
    # marinheiro: 35
    # preto: 62
    # verde: 62
    # TOTAL: 428


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
