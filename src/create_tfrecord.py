import sys
import argparse

from utils.tfrecords import load_dataset, save_tfrecords
from utils.data_reader import count_beans_in_list


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputdir', type=str, default='tests/images')
    parser.add_argument('-o', '--outputdir', type=str, default='tests/tfrecords')
    parser.add_argument('--im_size', type=int, default=64)
    parser.add_argument('--train_percent', type=float, default=0.8)
    parser.add_argument('--no-shuffle', dest='random', action='store_false', default=True)

    args = parser.parse_args()

    train_dataset, valid_dataset, teste_dataset = load_dataset(
        input_dir=args.inputdir,
        im_size=args.im_size,
        training_percentage=args.train_percent,
        random=args.random,
        n_files=(1, 1, 1)
    )

    print(f'{len(train_dataset)} train images')
    count_beans_in_list(train_dataset)

    print(f'{len(valid_dataset)} valid images')
    count_beans_in_list(valid_dataset)

    print(f'{len(teste_dataset)} teste images')
    count_beans_in_list(teste_dataset)

    save_tfrecords(train_dataset, 'train_dataset', args.outputdir, n=1)
    save_tfrecords(valid_dataset, 'valid_dataset', args.outputdir, n=1)
    save_tfrecords(teste_dataset, 'teste_dataset', args.outputdir, n=1)

    print('Finished.')

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
