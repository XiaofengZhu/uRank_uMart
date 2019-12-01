./bash.sh
CUDA_VISIBLE_DEVICES=1 nohup python main.py --loss_fn urank --data_dir ${TF_RANK_DATA}/MSLR-WEB30K/1 --tfrecords_filename MSLR-WEB30K.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urank --data_dir ${TF_RANK_DATA}/MSLR-WEB30K/1 --tfrecords_filename MSLR-WEB30K.tfrecords >> MSLR-WEB30K_urank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=1 nohup python main.py --loss_fn urank --data_dir ${TF_RANK_DATA}/MSLR-WEB30K/2 --tfrecords_filename MSLR-WEB30K.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urank --data_dir ${TF_RANK_DATA}/MSLR-WEB30K/2 --tfrecords_filename MSLR-WEB30K.tfrecords >> MSLR-WEB30K_urank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=1 nohup python main.py --loss_fn urank --data_dir ${TF_RANK_DATA}/MSLR-WEB30K/3 --tfrecords_filename MSLR-WEB30K.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urank --data_dir ${TF_RANK_DATA}/MSLR-WEB30K/3 --tfrecords_filename MSLR-WEB30K.tfrecords >> MSLR-WEB30K_urank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=1 nohup python main.py --loss_fn urank --data_dir ${TF_RANK_DATA}/MSLR-WEB30K/4 --tfrecords_filename MSLR-WEB30K.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urank --data_dir ${TF_RANK_DATA}/MSLR-WEB30K/4 --tfrecords_filename MSLR-WEB30K.tfrecords >> MSLR-WEB30K_urank.txt
./bash.sh
CUDA_VISIBLE_DEVICES=1 nohup python main.py --loss_fn urank --data_dir ${TF_RANK_DATA}/MSLR-WEB30K/5 --tfrecords_filename MSLR-WEB30K.tfrecords
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py --loss_fn urank --data_dir ${TF_RANK_DATA}/MSLR-WEB30K/5 --tfrecords_filename MSLR-WEB30K.tfrecords >> MSLR-WEB30K_urank.txt

echo "\n" >> MSLR-WEB30K_urank.txt
cat experiments/base_model/params.json >> MSLR-WEB30K_urank.txt