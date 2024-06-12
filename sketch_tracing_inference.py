import os
import six
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_WARNINGS"] = "0"

import model_full as sketch_tracing_model
from utils import load_dataset


def main(dataset_base, data_type, img_seq, mode='inference'):
    model_params = sketch_tracing_model.get_default_hparams()
    model_params.add_hparam('dataset_base', dataset_base)
    model_params.add_hparam('data_type', data_type)
    model_params.add_hparam('img_seq', img_seq)

    print('Hyperparams:')
    for key, val in six.iteritems(model_params.values()):
        print('%s = %s' % (key, str(val)))
    print('-' * 100)

    val_set = load_dataset(model_params)
    model = sketch_tracing_model.FullModel(model_params, val_set)

    if mode == 'inference':
        inference_root = model_params.inference_root
        sub_inference_root = os.path.join(inference_root, model_params.data_type)
        os.makedirs(sub_inference_root, exist_ok=True)
        model.inference(sub_inference_root, model_params.img_seq)
    # elif mode == 'test':
    #     model.evaluate(load_trained_weights=True)
    else:
        raise Exception('Unknown mode:', mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_base', '-db', type=str, default='sample_inputs', help="define the data base")
    parser.add_argument('--data_type', '-dt', type=str, default='rough', choices=['clean', 'rough'], help="define the data type")
    parser.add_argument('--img_seq', '-is', type=str, default=['23-0.png', '23-1.png', '23-2.png'], nargs='+', help="define the image sequence")

    args = parser.parse_args()

    main(args.dataset_base, args.data_type, args.img_seq)
