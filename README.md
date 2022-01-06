# semantic-segmentation
## setup
Prepare the data, and rewrite the config file to specify the path to the data. (default: semantic-segmentation/data)

## Training

```bash
python3 train.py [-c] [-d] 
```

e.g.
```bash
python3 train.py -c config/train_deeplabv3.json -d results/deeplabv3_res50
```


## Testing

```bash
python3 eval.py [-d] [-o]
```

e.g.
```
python3 eval.py -d results/deeplabv3_res50 -o results/deeplabv3_res50/sub
```

