./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn grank --data_dir ../data/MSLR-WEB30K/1 --tfrecords_filename MSLR-WEB30K.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn grank --data_dir ../data/MSLR-WEB30K/1 --tfrecords_filename MSLR-WEB30K.tfrecords >> MSLR-WEB30K_gRank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn grank --data_dir ../data/MSLR-WEB30K/2 --tfrecords_filename MSLR-WEB30K.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn grank --data_dir ../data/MSLR-WEB30K/2 --tfrecords_filename MSLR-WEB30K.tfrecords >> MSLR-WEB30K_gRank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn grank --data_dir ../data/MSLR-WEB30K/3 --tfrecords_filename MSLR-WEB30K.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn grank --data_dir ../data/MSLR-WEB30K/3 --tfrecords_filename MSLR-WEB30K.tfrecords >> MSLR-WEB30K_gRank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn grank --data_dir ../data/MSLR-WEB30K/4 --tfrecords_filename MSLR-WEB30K.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn grank --data_dir ../data/MSLR-WEB30K/4 --tfrecords_filename MSLR-WEB30K.tfrecords >> MSLR-WEB30K_gRank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=0 nohup python main.py --loss_fn grank --data_dir ../data/MSLR-WEB30K/5 --tfrecords_filename MSLR-WEB30K.tfrecords
CUDA_VISIBLE_DEVICES=0 nohup python evaluate.py --loss_fn grank --data_dir ../data/MSLR-WEB30K/5 --tfrecords_filename MSLR-WEB30K.tfrecords >> MSLR-WEB30K_gRank.txt
