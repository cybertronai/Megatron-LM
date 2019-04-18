export valid=/ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00099

# files contains list of all tfrecords in /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded
export files=`cat ~/bookcorpus/final_tfrecords_sharded/files.txt`

python ~/Megatron-LM/pretrain_bert.py     --batch-size 4     --tokenizer-type BertWordPieceTokenizer     --cache-dir cache_dir     --tokenizer-model-type bert-large-uncased     --vocab-size 30522     --use-tfrecords     --train-data $files     --valid-data $valid     --test-data $valid     --max-preds-per-seq 80     --seq-length 512     --max-position-embeddings 512     --num-layers 12     --hidden-size 768     --intermediate-size 4096     --num-attention-heads 12     --hidden-dropout 0.1     --attention-dropout 0.1     --train-iters 1000000     --lr 0.0001     --lr-decay-style linear     --lr-decay-iters 990000     --warmup .01     --weight-decay 1e-2     --clip-grad 1.0     --fp16     --fp32-layernorm     --fp32-embedding     --hysteresis 2     --run_name 110M
