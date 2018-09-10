# SegToLabelmeJSON

This script is to create JSON of labelme from segmentaion mask images.

## How to use

1. Prepare original image and segmentaion mask images according to the following directory structure.

```
[directory structure]

/
 seg_to_labelme_JSON.py
 data/
      images/  # original image
      masks/   # segmentaion mask images
```

   Also, make filename of segmentaion mask images `{original-image-name}_{label_name}_{No.}.png`.

```
[Exsample]

original image : 00001.png
egmentaion mask images : 00001_cat_0.png
                         00001_cat_1.png
                         00001_cat_2.png
                         00001_dog_0.png
                         ...
```


2. Execute seg_to_labelme_JSON.py

```
$ python seg_to_labelme_JSON.py
```

   JSON of labelme will be named `{original-image-name}.json`, and created in the same directory as original image.
