#!/usr/bin/env python
# forked from launch_8gpu.py

import argparse
import ncluster

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='pretrain_bert',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--instance_type', type=str, default="p3.2xlarge",
                    help="which instance type to use")
parser.add_argument('--image_name', type=str,
                    default='Deep Learning AMI (Ubuntu) Version 22.0',
                    help="name of AMI to use ")
parser.add_argument('--fp16', action="store_true",
                    help='enable fp16 training')
args = parser.parse_args()

ncluster.set_backend('aws')


def main():
    task = ncluster.make_task(name=args.name,
                              run_name=f"{args.name}",
                              image_name=args.image_name,
                              instance_type=args.instance_type)

    task.upload('*')
    task.run('killall python || echo failed')  # kill previous run
    task.run('source activate pytorch_p36')
    task.run('pip install -r requirements.txt')
    # workaround for https://github.com/tensorflow/models/issues/3995
    task.run('pip install -U protobuf')

    train = open('bookcorpus.filelist.train').read().strip()
    validate = "/ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000163"
    test = "/ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000164"

    lr = 0.0001  # original learning rate for 256 global batch size/64 GPUs
    lr = lr / (256 / 15)

    cmd = (f"python pretrain_bert.py "
           f"--batch-size 5 "
           f"--tokenizer-type BertWordPieceTokenizer "
           f"--cache-dir cache_dir "
           f"--tokenizer-model-type bert-large-uncased "
           f"--vocab-size 30522 "
           f"--use-tfrecords "
           f"--train-data {train} "
           f"--valid-data {validate} "
           f"--test-data {test} "
           f"--max-preds-per-seq 80 "
           f"--seq-length 512 "
           f"--max-position-embeddings 512 "
           f"--num-layers 16 "
           f"--hidden-size 410 "
           f"--intermediate-size 4096 "
           f"--num-attention-heads 10 "
           f"--hidden-dropout 0.1 "
           f"--attention-dropout 0.1 "
           f"--train-iters 1000000 "
           f"--lr {lr} "
           f"--lr-decay-style linear "
           f"--lr-decay-iters 990000 "
           f"--warmup .01 "
           f"--weight-decay 1e-2 "
           f"--clip-grad 1.0 "
           f"--fp32-layernorm "
           f"--fp32-embedding "
           f"--hysteresis 2 "
           f"--num-workers 2 ")

    # new params
    cmd += f"--logdir {task.logdir} "

    if args.fp16:
        cmd += f"--fp16 "

    task.run(f'echo {cmd} > {task.logdir}/task.cmd')  # save command-line
    task.run(cmd, non_blocking=True)

    print(f"Logging to {task.logdir}")


if __name__ == '__main__':
    main()
