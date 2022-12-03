<strong>VeeP:</strong><br />
In this repository, we provide an implementation for the paper "Boosting Robustness Verification of Semantic Feature Neighborhoods" https://arxiv.org/abs/2209.05446. The repository owner is anan.kabaha@campus.technion.ac.il. 

<strong>To get the models and the samples:</strong><br />
cd ./data/<br />
bash get_samples.sh<br />
cd ../models/<br />
bash get_models.sh<br />
cd ..

<strong>VeeP paramters:</strong><br />
--netname: the network name, the extension can be only .onnx<br />
--dataset: the dataset, can be either mnist, cifar10, or imagenet<br />
--max_itr: maximal number of iterations<br />
--timeout: timeout in seconds<br />
--M: history length<br />
--pertubration_type: perturbation type can be either brightness, saturation, hue, lightness, or brightness_and_contrast.<br />
--gpus: list of gpus to use<br />
--output_file: the file to save the results into<br />

<strong>Example:</strong><br />
python3 ./run.py --netname ./models/MNIST_10x5000.onnx --dataset mnist --gpus 0,1,2,3,4,5,6,7 
