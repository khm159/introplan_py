# introplan_py


## Notes 

- Original Repo : [https://github.com/kevinliang888/IntroPlan](https://github.com/kevinliang888/IntroPlan)

- Original Paper : [Kaiqu Liang, Zixu Zhang, Jaime Fernández Fisac. et al., "Introspective Planning: Aligning Robots’ Uncertainty with Inherent Task Ambiguity."](https://arxiv.org/abs/2402.06529)

I convert original peper's jupyter notebook style codes to python codes due to enhance the usability.

Currently, Llama is only inferencible

I will modify this code to support several huggingface models such as Phi-3 and openai models.

And i modify requirements.txt. Because there are missing component in the original github repo. 


## Converted python file 

- Mobile Manipulation Conformal Prediction Safe 

```bash
python mobile_safe_cp.py
```

## Environment 

```bash
conda create -n intropy python=3.10 -y
conda activate intropy
pip install -r requirements.txt
```




