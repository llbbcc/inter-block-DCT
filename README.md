# Inter-block-DCT
An unofficial code implementation of 'Robust and blind image watermarking in DCT domain using inter-block coefficient correlation'

## Embed watermark

```bash
python main_class.py --options embed --shuffle True
```

## Attack

```bash
python main_class.py --options attack --quality 80 --attack quality
```

## Extract watermark

```bash
python main_class.py --options extract --root_wtm_img ./attack/masked_img.jpg --shuffle True
```
