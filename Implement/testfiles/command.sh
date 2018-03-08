#train
#python /home/carlchang/YeNetTensorflow/Implement/main.py /home/carlchang/YeNetTensorflow/Experiment/TryCaenorstStructure/train/cover /home/carlchang/YeNetTensorflow/Experiment/TryCaenorstStructure/train/stego /home/carlchang/YeNetTensorflow/Experiment/TryCaenorstStructure/valid/cover /home/carlchang/YeNetTensorflow/Experiment/TryCaenorstStructure/valid/stego --log-path="/home/carlchang/YeNetTensorflow/Experiment/TryCaenorstStructure/log" --use-batch-norm --lr 4e-1 --max-epochs=300 --log-interval=24 --gpu="0,1,2,3"

#test
#python /home/carlchang/YeNetTensorflow/Implement/main.py /home/carlchang/YeNetTensorflow/Experiment/TryCaenorstStructure/test/cover /home/carlchang/YeNetTensorflow/Experiment/TryCaenorstStructure/test/stego --log-path="/home/carlchang/YeNetTensorflow/Experiment/TryCaenorstStructure/log" --gpu="0,1,2,3"

#data_split
python /home/carlchang/YeNetTensorflow/Implement/main.py /home/carlchang/YeNetTensorflow/Data/SUNI_0.4_15000 /home/carlchang/YeNetTensorflow/Experiment/SUNI_0.4_15000_No_2 --train-percent=0.6 --valid-percent=0.2 --test-percent=0.2 --gpu=" "
