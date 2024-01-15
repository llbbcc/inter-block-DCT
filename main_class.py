import numpy as np
import cv2
from numpy.random.mtrand import shuffle
from skimage import metrics
import argparse
from utils import calculate_diff, calculate_M, f16, f4, f4_test

parser = argparse.ArgumentParser()
parser.add_argument('--root_img', type=str, default="../fig/Lena_512x512.jpg", help='root of image')
parser.add_argument('--root_wtm', type=str, default='../fig/watermark.png', help='root of watermark')
parser.add_argument('--root_wtm_img', type=str, default='./embed/watermarked_img.jpg', help='root of watermarked image')
parser.add_argument('--block_size', type=int, default=8, help='size of each block for dct')
parser.add_argument('--Z', type=int, default=2, help='')
parser.add_argument('--Th', type=int, default=80, help='threshold for part ranges')
parser.add_argument('--K', type=int, default=12, help='robust factor')
parser.add_argument('--quality', type=int, default=80, help='quality of JPEG compress')
parser.add_argument('--shuffle', type=bool, default=False, help='shuffle watermark or not')
parser.add_argument('--attack', type=str, default='quality', help='attack options')
parser.add_argument('--options', type=str, choices=['embed', 'extract', 'attack', 'metrics'])
args = parser.parse_args()

class INTER_DCT(object):
    def __init__(self,
                image,
                watermark,
                block_size=8, 
                Z=2, 
                Th=80, 
                K=12, 
                quality=80, 
                shuffle=False, 
                sample_size=384, 
                mask_loc = 'TR' # TL, BR, BL, B, T, L, R
                ) -> None:
        super().__init__()
        self.image = image
        self.watermark = watermark
        self.block_size = block_size
        self.Z = Z
        self.Th = Th
        self.K = K
        self.quality = quality
        self.shuffle = shuffle
        self.shuffled_wtm = watermark
        self.sample_size = sample_size
        self.mask_loc = mask_loc

    def watermark_embed(self):
        if self.shuffle:
            watermark_flatten = self.watermark.flatten()
            indices = np.arange(len(watermark_flatten))
            shuffled_indices = indices.copy()
            np.random.shuffle(shuffled_indices)
            np.save('shuffled_indices.npy', shuffled_indices)
            shuffled_watermark = watermark_flatten[shuffled_indices]
            self.shuffled_wtm = np.resize(shuffled_watermark, (64, 64))
            # cv2.imwrite("./watermark.jpg", watermark * 255)
            watermarked_image = self.embedding()
        else:
            watermarked_image = self.embedding()
        cv2.imwrite('./embed/watermarked_img.jpg', watermarked_image)

    def watermark_extract(self, root):
        # Extraction example
        watermarked_image = cv2.imread(root, cv2.IMREAD_GRAYSCALE)
        watermarked_image = np.float32(watermarked_image)
        watermark = self.extract(watermarked_image)
        if self.shuffle:
            shuffled_indices = np.load('shuffled_indices.npy')
            watermark_flt = watermark.flatten()
            watermark_flt = watermark_flt[np.argsort(shuffled_indices)]
            watermark = np.resize(watermark_flt, (64, 64))
        cv2.imwrite("./extract/watermark.jpg", watermark)

    def dct2(self, img):
        return cv2.dct(np.float32(img))

    def idct2(self, img):
        return cv2.idct(np.float32(img))

    def embedding(self):
        img = self.image - 128
        rows, cols = img.shape
        R, C = rows // self.block_size, cols // self.block_size
        dct_blocks = np.zeros((R, C, self.block_size, self.block_size))

        for i in range(R):
            for j in range(C):
                block = img[i*self.block_size:(i+1)*self.block_size, j*self.block_size:(j+1)*self.block_size]
                dct_blocks[i, j] = self.dct2(block)
        
        for i in range(R):
            for j in range(C):
                # Compute median and DC values
                zigzag = np.concatenate([np.diagonal(dct_blocks[i, j][::-1, :], k)[::(2 * (k % 2) - 1)] for k in range(1 - self.block_size, self.block_size)])
                Med = np.median(zigzag[1:10])
                DC = dct_blocks[i, j, 0, 0]

                # Compute modification parameter
                if abs(DC) > 1000 or abs(DC) < 1:
                # elif abs(DC) > 1000 or abs(DC) < 1 or abs(Med) > abs(DC):
                    M = self.Z * Med
                else:
                    M = self.Z * (DC - Med) / DC
                if self.shuffle:
                    watermark_bit = self.shuffled_wtm[i, j]
                else:
                    watermark_bit = self.watermark[i, j]

                x, y = 3, 3
                # Calculate Diff for each direction
                if i % 2 == 0 and j % 2 == 0:
                    direction = 'LR'
                elif i % 2 == 1 and j % 2 == 0:
                    direction = 'DU'
                elif i % 2 == 0 and j % 2 == 1:
                    direction = 'UD'
                else:
                    direction = 'RL'
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)

                # Embed watermark based on the Diff and watermark bit
                f4(dct_blocks, watermark_bit, Diff, self.Th, self.K, i, j, self.Z, self.block_size, x, y, direction, M)
                # Apply inverse DCT to modified block
                watermarked_block = self.idct2(dct_blocks[i, j])
                # print(np.max(watermarked_block))
                img[i*self.block_size:(i+1)*self.block_size, j*self.block_size:(j+1)*self.block_size] = watermarked_block
        
        img += 128

        return img
    
    def extract(self, watermarked_img):
        img = watermarked_img - 128

        # Extract watermark bit by bit
        rows, cols = img.shape
        R, C = rows // self.block_size, cols // self.block_size
        dct_blocks = np.zeros((R, C, self.block_size, self.block_size))
        extracted_watermark = np.zeros((R, C))

        # Perform DCT for each block and store in dct_blocks
        for i in range(R):
            for j in range(C):
                block = img[i*self.block_size:(i+1)*self.block_size, j*self.block_size:(j+1)*self.block_size]
                dct_blocks[i, j] = self.dct2(block)

        for i in range(R):
            for j in range(C):
                # Current DCTed block
                block = dct_blocks[i, j]
                # Initialize direction
                if i % 2 == 0 and j % 2 == 0:
                    direction = 'LR'
                elif i % 2 == 1 and j % 2 == 0:
                    direction = 'DU'
                elif i % 2 == 0 and j % 2 == 1:
                    direction = 'UD'
                else:
                    direction = 'RL'
                x, y = 3, 3
                Diff, result = calculate_diff(dct_blocks, i, j, x, y, direction)

                # if Diff > 7 * Th or (Diff > -7 * Th and abs(Diff // Th) % 2 == 1):
                if Diff < -self.Th or (0 <= Diff < self.Th):
                    extracted_watermark[i, j] = 1
                else:
                    extracted_watermark[i, j] = 0
                    
        return extracted_watermark * 255

    def attack(self, root, options):
        img = cv2.imread(root, cv2.IMREAD_GRAYSCALE)
        if options == 'crop':
            h, w = img.shape[0], img.shape[1]
            img = img[:9 * h // 10, : 9 * w // 10]
            img = cv2.resize(img, (h, w))
            cv2.imwrite('./attack/croped_img.jpg', img)
        elif options == 'mask':
            mask = np.ones(img.shape[:2], dtype="uint8")*255
            h, w = img.shape[0], img.shape[1]
            if self.mask_loc == 'TR':
                mask[:h // 2, w // 2:] = 0
            elif self.mask_loc == 'TL':
                mask[:h // 2, :w // 2] = 0
            elif self.mask_loc == 'BR':
                mask[h // 2:, w // 2:] = 0
            elif self.mask_loc == 'BL':
                mask[h // 2:, :w // 2] = 0
            elif self.mask_loc == 'B':
                mask[h // 2:, :] = 0
            elif self.mask_loc == 'T':
                mask[:h // 2, :] = 0
            elif self.mask_loc == 'L':
                mask[:, : w // 2]
            elif self.mask_loc == 'R':
                mask[:, w // 2:]
            else:
                raise KeyError('Unknown mask location')
            img = cv2.bitwise_and(img, img, mask=mask)
            cv2.imwrite('./attack/masked_img.jpg', img)
        elif options == 'quality':
            cv2.imwrite('./attack/compressed_quality_{}.jpg'.format(self.quality), img , [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        elif options == 'dct':
            h, w = img.shape[0], img.shape[1]
            R, C = h // self.block_size, w // self.block_size

            dct_blocks = np.zeros((R, C, self.block_size, self.block_size))

            for i in range(R):
                for j in range(C):
                    block = img[i*self.block_size:(i+1)*self.block_size, j*self.block_size:(j+1)*self.block_size]
                    dct_blocks[i, j] = self.dct2(block).astype(np.float16)
            results = np.zeros_like(img)
            for i in range(R):
                for j in range(C):
                    watermarked_img = self.idct2(dct_blocks[i, j])
                    results[i*self.block_size:(i+1)*self.block_size, j*self.block_size:(j+1)*self.block_size] = watermarked_img
            cv2.imwrite('./attack/compressed_dct.jpg', results)
        elif options == 'linear':
            h, w = img.shape[0], img.shape[1]
            img = cv2.resize(img, (self.sample_size, self.sample_size), interpolation=cv2.INTER_LINEAR)
            img = cv2.resize(img, (h, w))
            cv2.imwrite('./attack/sample_linear_{}.jpg'.format(self.sample_size), img)
        elif options == 'cubic':
            h, w = img.shape[0], img.shape[1]
            img = cv2.resize(img, (self.sample_size, self.sample_size), interpolation=cv2.INTER_CUBIC)
            img = cv2.resize(img, (h, w))
            cv2.imwrite('./attack/sample_cubic_{}.jpg'.format(self.sample_size), img)
        else:
            raise KeyError('Unknown options!')

class Metric():
    def __init__(self) -> None:
        pass

    def PSNR(self, img1, img2, data_range=255):
        return metrics.peak_signal_noise_ratio(img1, img2, data_range=data_range)

    def SSIM(self, img1, img2):
        return metrics.structural_similarity(img1, img2, full=True, win_size=7)
    
    def minus(self, img1, img2):
        img = img1 - img2
        cv2.imwrite('results/minus.jpg',img)

if __name__ == '__main__':
    image = cv2.imread(args.root_img, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    image = np.float32(image)
    watermark = cv2.imread(args.root_wtm, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.resize(watermark, (64, 64))
    watermark = np.where(watermark < 128, 0, 1)

    inter_dct = INTER_DCT(
        image=image,
        watermark=watermark,
        block_size=args.block_size,
        Z=args.Z,
        Th=args.Th,
        K=args.K,
        quality=args.quality,
        shuffle=args.shuffle,
    )
    me = Metric()
    if args.options == 'embed':
        inter_dct.watermark_embed()
    elif args.options == 'extract':
        inter_dct.watermark_extract(args.root_wtm_img)
    elif args.options == 'attack':
        inter_dct.attack(args.root_wtm_img, args.attack)
    else:
        ori_img = cv2.imread(args.root_img, cv2.IMREAD_GRAYSCALE)
        ori_img = cv2.resize(ori_img, (512, 512))
        wtm_img = cv2.imread(args.root_wtm_img, cv2.IMREAD_GRAYSCALE)
        wtm_img = cv2.resize(wtm_img, (512, 512))
        minus = me.minus(ori_img, wtm_img)
        ssim = me.SSIM(ori_img, wtm_img)
        print('PSNR:', me.PSNR(ori_img, wtm_img))

