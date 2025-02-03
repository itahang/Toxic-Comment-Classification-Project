# Toxic Comment Classification



How to use it?
1. Git clone it 

    ```shell
    git clone github.com/itahang/Toxic-Comment-Classification-Project
    ```

2. Install the requirements 

    ```shell
    pip install -r requirements.txt
    ```

3. You have to install JAX for that read the **Note** Section
4. Run the app  
    ```shell
    streamlit run Main.py
    ```

5. It will open at some localhost port

## Note:

If you dont have GPU perfrom this:
```shell
pip install jax
```

If you have GPU then use this command:
```shell
pip install -U "jax[cuda12]"
```

By time of this writing JAX's GPU version only supports `Linux x86_64` and
`Linux aarch64` . In Windows its Experimental [Details](https://jax.readthedocs.io/en/latest/installation.html)


So in cause its does not install properly use CPU version insted
We use WSL for CUDA support and its works properly on WSL 
So if you want to use GPU version use WSL

