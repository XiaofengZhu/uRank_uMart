./bash.sh
CUDA_VISIBLE_DEVICES=1 python main.py --loss_fn urrank --data_dir ../data/OHSUMED/1 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ../data/OHSUMED/1 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_urRank.txt

./bash.sh
CUDA_VISIBLE_DEVICES=1 python main.py --loss_fn urrank --data_dir ../data/OHSUMED/2 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ../data/OHSUMED/2 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_urRank.txt

./bash.sh
CUDA_VISIBLE_DEVICES=1 python main.py --loss_fn urrank --data_dir ../data/OHSUMED/3 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ../data/OHSUMED/3 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_urRank.txt

./bash.sh
CUDA_VISIBLE_DEVICES=1 python main.py --loss_fn urrank --data_dir ../data/OHSUMED/4 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ../data/OHSUMED/4 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_urRank.txt

./bash.sh
CUDA_VISIBLE_DEVICES=1 python main.py --loss_fn urrank --data_dir ../data/OHSUMED/5 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urrank --data_dir ../data/OHSUMED/5 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_urRank.txt