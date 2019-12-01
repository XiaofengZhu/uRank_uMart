./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn grank --data_dir ../data/OHSUMED/1 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn grank --data_dir ../data/OHSUMED/1 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_gRank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn grank --data_dir ../data/OHSUMED/2 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn grank --data_dir ../data/OHSUMED/2 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_gRank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn grank --data_dir ../data/OHSUMED/3 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn grank --data_dir ../data/OHSUMED/3 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_gRank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn grank --data_dir ../data/OHSUMED/4 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn grank --data_dir ../data/OHSUMED/4 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_gRank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn grank --data_dir ../data/OHSUMED/5 --tfrecords_filename OHSUMED.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn grank --data_dir ../data/OHSUMED/5 --tfrecords_filename OHSUMED.tfrecords >> OHSUMED_gRank.txt