# Annotation-Tool
A tool to annotate images in a large dataset efficiently by incorporating Machine Learning techniques.

We developed an offline mobile web app (local setup available) that is designed primarily to annotate unlabeled images. To minimize the cost of annotation (i.e number of images to be manually annotated), we use Active Learning for predicting the labels of the next set of images by an ML model which learns from the images manually annotated by a domain expert. 


# Project Goals:
* Fast Human Annotation
* Incorporating Active Learning
* File I/O consistency



## Setup (Done only the first time)
* Clone the repo. 
* Download Miniconda from here `https://docs.conda.io/en/latest/miniconda.html`
* Once downloaded open the exectuable File `Miniconda3-latest-Windows-x86_64`. And follow the usual installation process.
* After the installation gets completed open command-prompt and type `conda --version`. If you get a prompt saying: `conda 4.9.2`, you have correctly installed.
* Create a conda env: 
  ```bash
     conda create -y -n at37 python=3.7```

* Create an environment:
  - ```bash
    conda activate at37
    ```
* Navigate to the local folder where you have the cloned repo.
* Ensure you are present in **Seeds_Project/Ann_Tool_Seeds_Proj**
* Install dependencies
  ```bash
  pip install -r requirements.txt
  ```




#### Starting a session 
* ```bash
  python create_start_state.py --is_os_win 0 --initials hk --run 1 --global_reset 0 --img_dir_path ./static/Path2ImageFolder
  ```
* ```bash
  python main.py --is_os_win 0 --initials hk --img_dir_path ./static/Path2ImageFolder
  ```
* Copy everything after 'Dash is running on' say (http://127.0.0.1:7236) and open a new browser tab (say Chrome/Mozilla etc) and paste in the URL field of the tab.
