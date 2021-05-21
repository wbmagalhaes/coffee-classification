import sys
import argparse

from random import shuffle

from utils.data_reader import open_image, open_jsons
from utils.segmentation import crop_beans, count_beans_set
from utils.tfrecords import save_tfrecord


def load_datafiles(input_dir, im_size=64, random=True, train_percent=0.8, n_files=(1, 1, 1)):
    jsons, addrs = open_jsons(input_dir)

    dataset = []
    for data, addr in zip(jsons, addrs):
        image = open_image(addr[:-4] + 'jpg')
        beans = crop_beans(image, data, cut_size=im_size)
        dataset.extend(beans)

    if random:
        shuffle(dataset)

    train_num = int(len(dataset) * train_percent)
    teste_num = int(len(dataset) * (1 - train_percent)) // 2

    train = dataset[:train_num]
    valid = dataset[train_num:train_num + teste_num]
    teste = dataset[train_num + teste_num:]

    return train, valid, teste


def save_datasets(output_dir, train, valid, teste):
    save_tfrecord(train, 'train_dataset', output_dir, n=1)
    save_tfrecord(valid, 'valid_dataset', output_dir, n=1)
    save_tfrecord(teste, 'teste_dataset', output_dir, n=1)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdir', type=str, default='images')
    parser.add_argument('-o', '--outputdir', type=str, default='data')
    parser.add_argument('--im_size', type=int, default=64)
    parser.add_argument('--train_percent', type=float, default=0.8)
    parser.add_argument('--no-shuffle', dest='random', action='store_false', default=True)
    args = parser.parse_args()

    train, valid, teste = load_datafiles(
        input_dir=args.inputdir,
        im_size=args.im_size,
        train_percent=args.train_percent,
        random=args.random,
        n_files=(1, 1, 1)
    )

    print(f'{len(train)} train images')
    count_beans_set(train)
    print('')

    print(f'{len(valid)} valid images')
    count_beans_set(valid)
    print('')

    print(f'{len(teste)} teste images')
    count_beans_set(teste)
    print('')

    save_datasets(args.outputdir, train, valid, teste)

    print('Finished.')

    # ? ===== DATASET =====
    # sadio: 1149
    # ardido: 1139
    # brocado: 404
    # marinheiro: 307
    # preto: 615
    # verde: 661
    # TOTAL: 4275

    # ? ===== TREINAMENTO (80%) =====
    # sadio: 912
    # ardido: 910
    # brocado: 334
    # marinheiro: 241
    # preto: 500
    # verde: 523
    # TOTAL: 3420

    # ? ===== VALIDAÇÃO (10%) =====
    # sadio: 114
    # ardido: 114
    # brocado: 39
    # marinheiro: 31
    # preto: 53
    # verde: 76
    # TOTAL: 427

    # ? ===== TESTE (10%) =====
    # sadio: 123
    # ardido: 115
    # brocado: 31
    # marinheiro: 35
    # preto: 62
    # verde: 62
    # TOTAL: 428


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
