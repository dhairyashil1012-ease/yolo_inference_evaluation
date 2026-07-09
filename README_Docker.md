## Step 1: Clone the Repository

```bash
git clone docker_setup https://github.com/dhairyashil1012-ease/yolo_inference_evaluation.git

cd yolo_inference_evaluation
```




## Step 2: For TensorRT Docker Images


```https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/tensorrt/26.05-py3/tags```



### For Sample Input Images You can take yours or you can reference below link as well 
```bash
https://drive.google.com/drive/folders/1MuNF6ytZHTcroBpnlwIUoWkaxl0oTTOS
```
If you install sample images from above link extract this in each Evaluation directory 



## Step 3 : Mount Your Dir. and Run Image 
docker run  -it --gpus all -p 8888:8888 -v `pwd`:/workspace -w /workspace nvcr.io/nvidia/nvcr.io/nvidia/tensorrt:26.06-py3 bash




## Step 4: Install Dependencies

Install all required packages.

```bash
pip install -r requirements.txt
```




## Step 5 :- Installation 

```bash
apt-get install -y     libgl1     libglib2.0-0     libsm6     libxext6     libxrender1     libxcb1

```

### Step 6:- Jupyter-Notebook 
```bash
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
```


### Copy url and open in browser . Start Execution according to your use case. For more information Follow each particular Dir. README.md file
