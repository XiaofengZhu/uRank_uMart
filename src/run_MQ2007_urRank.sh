./bash.sh
CUDA_VISIBLE_DEVICES=2 nohup python main.py --loss_fn urrank --data_dir ../data/MQ2007/1 --tfrecords_filename MQ2007.tfrecords
CUDA_VISIBLE_DEVICES=2 nohup python evaluate.py --loss_fn urrank --data_dir ../data/MQ2007/1 --tfrecords_filename MQ2007.tfrecords >> MQ2007_urRank.txt

./bash.sh
CUDA_VISIBLE_DEVICES=2 nohup python main.py --loss_fn urrank  --data_dir ../data/MQ2007/2 --tfrecords_filename MQ2007.tfrecords
CUDA_VISIBLE_DEVICES=2 nohup python evaluate.py --loss_fn urrank  --data_dir ../data/MQ2007/2 --tfrecords_filename MQ2007.tfrecords >> MQ2007_urRank.txt

./bash.sh
CUDA_VISIBLE_DEVICES=2 nohup python main.py --loss_fn urrank --data_dir ../data/MQ2007/3 --tfrecords_filename MQ2007.tfrecords
CUDA_VISIBLE_DEVICES=2 nohup python evaluate.py --loss_fn urrank --data_dir ../data/MQ2007/3 --tfrecords_filename MQ2007.tfrecords >> MQ2007_urRank.txt

./bash.sh
CUDA_VISIBLE_DEVICES=2 nohup python main.py --loss_fn urrank --data_dir ../data/MQ2007/4 --tfrecords_filename MQ2007.tfrecords
CUDA_VISIBLE_DEVICES=2 nohup python evaluate.py --loss_fn urrank --data_dir ../data/MQ2007/4 --tfrecords_filename MQ2007.tfrecords >> MQ2007_urRank.txt

./bash.sh
CUDA_VISIBLE_DEVICES=2 nohup python main.py --loss_fn urrank --data_dir ../data/MQ2007/5 --tfrecords_filename MQ2007.tfrecords
CUDA_VISIBLE_DEVICES=2 nohup python evaluate.py --loss_fn urrank --data_dir ../data/MQ2007/5 --tfrecords_filename MQ2007.tfrecords >> MQ2007_urRank.txt