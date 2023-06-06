nohup python main.py --save_dir=Sports/our+ --gpu=7 --expand=9999 --dataset=Sports > out/sports+ &
nohup python main.py --save_dir=Sports/orgin --teaching_epoch=1000 --gpu=1 --dataset=Sports > out/sports &
nohup python main.py --save_dir=Yelp/our+ --expand=9999 --teaching_epoch=300 --gpu=7 --dataset=Yelp > out/yelp+ &
nohup python main.py --save_dir=Yelp/orgin --teaching_epoch=1000 --gpu=6 --dataset=Yelp > out/yelp &

