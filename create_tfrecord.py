import sys
import argparse

from utils.tfrecords import create_dataset


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputdir', type=str)
    parser.add_argument('-o', '--outputdir', type=str)
    parser.add_argument('--im_size', type=int, default=64)
    parser.add_argument('--train_percent', type=float, default=0.8)
    parser.add_argument('--no-shuffle', dest='random', action='store_false', default=True)

    args = parser.parse_args()

    create_dataset(
        input_dir=args.inputdir,
        output_dir=args.outputdir,
        im_size=args.im_size,
        training_percentage=args.train_percent,
        random=args.random,
        n_files=(1, 1, 1)
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
