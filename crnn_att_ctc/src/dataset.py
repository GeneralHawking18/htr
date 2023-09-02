import torch, glob, os
from PIL import Image
import torch.nn.functional as F

class OCR_Dataset(torch.utils.data.Dataset):
    alphabets = "-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ"
    int2chr = {i+1: v for i, v in enumerate(alphabets)}
    int2chr[0] = ""
    chr2int = {v: i for i, v in int2chr.items()}
    # neg_ones = None

    def __init__(self, 
        imgs_dir, 
        img_width, img_height,
        gt_txt_dir = None, transform = None, 
        mode = "train"
    ):
        self.imgs_dir = imgs_dir
        super().__init__()

        self.transform = transform
        self.gt_txt_dir = gt_txt_dir
        self.list_imgs = []
        self.max_label_length = 0
        self.img_width = img_width
        self.img_height = img_height
        self.mode = mode
        
        if self.gt_txt_dir:
            self.get_gt()
            self.neg_ones = torch.ones(self.max_label_length) * (-1)
        else:
            self.list_imgs = [[img_dir] for img_dir in sorted(
                glob.glob("{}/*.jpg".format(imgs_dir)) + \
                glob.glob("{}/*.png".format(imgs_dir))
            )]

    def __len__(self):
        return len(self.list_imgs)

    @staticmethod
    def label_to_num(label):
        label_num = []
        for ch in label:
            label_num.append(OCR_Dataset.chr2int[ch])
        return torch.LongTensor(label_num)

    @staticmethod
    def num_to_labels(num):
        ret = ""
        for n in num:
            if n == 0:  # CTC Blank
                print(ret)
                print(num)
                break
            else:
                ret+= OCR_Dataset.int2chr[n]
        return ret

    def get_gt(self):
        with open(self.gt_txt_dir) as f:
            while True:
                # Get next line from file
                line = f.readline()

                if not line:
                    break
                try:
                    f_name, label = line.split()
                except ValueError:
                    raise Exception("Error in: " + line)

                img_dir = self.imgs_dir + "/" + f_name
                if self.max_label_length < len(label):
                    self.max_label_length = len(label)
                self.list_imgs.append((img_dir, label))


    def __getitem__(self, idx):
        img_dir = self.list_imgs[idx][0]
        img_name = os.path.basename(img_dir)

        img = Image.open(img_dir).convert("L").resize((self.img_width, self.img_height))# .transpose(Image.TRANSPOSE)
        # print(img.size)
        # display(img)
        if self.transform:
            img = self.transform(img)

        if self.gt_txt_dir: # if having grouth truth file, then having label
            label = self.list_imgs[idx][1]
            token = OCR_Dataset.label_to_num(label)
            targets = F.pad(token, pad=(0, self.max_label_length -len(token)), mode='constant', value=0).type('torch.IntTensor')
            #torch.zeros(size = self.max_label_length)
            # torch.full(size = (self.max_label_length, ), fill_value = len(self.alphabets))

            # targets = torch.clone(self.neg_ones)
            target_length = len(token)


            return img, targets, torch.IntTensor([target_length]), label

        else:
            return img_name, img
