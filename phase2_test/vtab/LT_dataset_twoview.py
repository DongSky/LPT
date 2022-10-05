from tokenize import group
import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image

places_mapping = {7: 0, 9: 0, 12: 0, 19: 0, 24: 0, 42: 0, 45: 0, 48: 0, 52: 0, 62: 0, 66: 0, 67: 0, 81: 0, 82: 0, 84: 0, 86: 0, 90: 0, 91: 0, 94: 0, 97: 0, 106: 0, 110: 1, 121: 1, 132: 1, 150: 1, 151: 1, 154: 1, 164: 1, 171: 1, 175: 1, 183: 1, 192: 1, 203: 1, 204: 1, 205: 1, 214: 1, 215: 1, 217: 1, 220: 1, 224: 1, 232: 1, 243: 1, 245: 1, 252: 1, 257: 1, 266: 1, 271: 1, 272: 1, 278: 1, 279: 1, 284: 1, 288: 1, 299: 1, 307: 1, 312: 1, 313: 1, 317: 1, 323: 1, 324: 1, 326: 1, 330: 1, 334: 1, 337: 1, 342: 1, 344: 1, 345: 1, 355: 1, 4: 1, 319: 1, 258: 1, 15: 1, 0: 1, 277: 1, 357: 1, 341: 1, 268: 1, 163: 1, 209: 1, 361: 1, 348: 1, 76: 1, 40: 1, 286: 1, 18: 1, 182: 1, 283: 1, 29: 1, 292: 1, 244: 1, 305: 1, 118: 1, 254: 1, 301: 1, 157: 1, 193: 1, 153: 1, 109: 1, 360: 1, 30: 1, 309: 1, 293: 1, 273: 1, 122: 1, 125: 1, 38: 1, 290: 1, 134: 1, 49: 1, 78: 1, 79: 1, 57: 1, 77: 1, 117: 1, 259: 1, 35: 1, 142: 1, 108: 1, 270: 1, 2: 1, 231: 1, 304: 1, 306: 1, 8: 1, 127: 1, 308: 1, 199: 1, 140: 1, 223: 1, 58: 1, 310: 1, 362: 1, 149: 1, 112: 1, 194: 1, 17: 1, 39: 1, 59: 1, 181: 1, 221: 1, 103: 1, 107: 1, 197: 1, 314: 1, 249: 1, 26: 1, 43: 1, 349: 1, 32: 1, 69: 1, 356: 1, 226: 1, 276: 1, 85: 1, 13: 1, 120: 1, 238: 1, 3: 1, 123: 1, 141: 1, 27: 1, 136: 1, 63: 1, 353: 1, 213: 1, 147: 1, 161: 1, 287: 1, 152: 1, 21: 1, 33: 1, 105: 1, 104: 1, 113: 1, 338: 1, 44: 1, 180: 1, 31: 1, 95: 1, 99: 1, 119: 1, 144: 1, 176: 1, 315: 1, 92: 1, 251: 1, 275: 1, 359: 1, 227: 1, 158: 1, 250: 1, 212: 1, 115: 1, 269: 1, 167: 1, 186: 1, 234: 1, 274: 1, 54: 1, 143: 1, 239: 1, 210: 1, 25: 1, 233: 1, 200: 1, 236: 1, 98: 1, 206: 1, 321: 1, 265: 1, 328: 1, 237: 1, 263: 1, 190: 2, 333: 2, 22: 2, 87: 2, 216: 2, 340: 2, 60: 2, 138: 2, 230: 2, 350: 2, 352: 2, 320: 2, 74: 2, 354: 2, 53: 2, 303: 2, 179: 2, 184: 2, 16: 2, 61: 2, 202: 2, 47: 2, 124: 2, 316: 2, 11: 2, 189: 2, 289: 2, 297: 2, 89: 2, 20: 2, 41: 2, 102: 2, 145: 2, 281: 2, 285: 2, 130: 2, 73: 2, 96: 2, 146: 2, 174: 2, 178: 2, 267: 2, 329: 2, 5: 2, 10: 2, 101: 2, 148: 2, 260: 2, 347: 2, 133: 2, 325: 2, 71: 2, 280: 2, 80: 2, 253: 2, 116: 2, 156: 2, 165: 2, 225: 2, 327: 2, 240: 2, 6: 2, 196: 2, 218: 2, 114: 2, 131: 2, 336: 2, 65: 2, 191: 2, 36: 2, 139: 2, 173: 2, 242: 2, 296: 2, 187: 2, 358: 2, 1: 2, 55: 2, 70: 2, 162: 2, 185: 2, 364: 2, 72: 2, 298: 2, 311: 2, 318: 2, 28: 2, 211: 2, 247: 2, 177: 2, 208: 2, 229: 2, 335: 2, 219: 2, 282: 2, 343: 2, 159: 2, 228: 2, 241: 2, 264: 2, 51: 2, 64: 2, 256: 2, 291: 2, 339: 2, 363: 2, 170: 2, 198: 2, 201: 2, 14: 2, 207: 2, 235: 2, 246: 2, 322: 2, 50: 2, 75: 2, 83: 2, 294: 2, 300: 2, 302: 2, 332: 2, 34: 2, 169: 2, 331: 2, 166: 2, 168: 2, 172: 2, 255: 2, 261: 2, 262: 2, 46: 2, 68: 2, 111: 2, 129: 2, 160: 2, 248: 2, 346: 2, 23: 2, 135: 2, 155: 2, 222: 2, 351: 2, 37: 2, 56: 2, 93: 2, 100: 2, 128: 2, 137: 2, 188: 2, 295: 2, 88: 2, 126: 2, 195: 2}

class LT_Dataset_twoview(Dataset):
    
    def __init__(self, root, txt, transform=None, template=None, top_k=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        # select top k class
        if top_k:
            # only select top k in training, in case train/val/test not matching.
            if 'train' in txt:
                max_len = max(self.labels) + 1
                dist = [[i, 0] for i in range(max_len)]
                for i in self.labels:
                    dist[i][-1] += 1
                dist.sort(key = lambda x:x[1], reverse=True)
                # saving
                torch.save(dist, template + '_top_{}_mapping'.format(top_k))
            else:
                # loading
                dist = torch.load(template + '_top_{}_mapping'.format(top_k))
            selected_labels = {item[0]:i for i, item in enumerate(dist[:top_k])}
            # replace original path and labels
            self.new_img_path = []
            self.new_labels = []
            for path, label in zip(self.img_path, self.labels):
                if label in selected_labels:
                    self.new_img_path.append(path)
                    self.new_labels.append(selected_labels[label])
            self.img_path = self.new_img_path
            self.labels = self.new_labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample1 = self.transform(image)
            sample2 = self.transform(image)
        
        # group_label = places_mapping[label]

        # return sample, label#, index
        return sample1, sample2, label, label
