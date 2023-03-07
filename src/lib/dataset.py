"""Datasets."""
import torch, glob, json, os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
import torch.nn.functional as F
import numpy as np
from PIL import Image


PROMPT_TEMPLATES_LARGE = {
    "1": "A photo of {}",
    "2": "A photo of the {}",
    "3": "A good photo of {}",
    "4": "A cropped photo of {}",
    "5": "A close-up photo of {}",
    "6": "A bright photo of {}",
    "7": "A face photo of {}",
    "8": "The beautiful face photo of {}",
    "9": "The photo of {}",
    "10": "The photo of the {}",
    "11": "The good photo of {}",
    "12": "The cropped photo of {}",
    "13": "The close-up photo of {}",
    "14": "The bright photo of {}",
    "15": "The face photo of {}",
    "16": "The beautiful face photo of {}",
}


PROMPT_TEMPLATES_MEDIUM = {
    "person": "A face photo of {}",
    "smile": "A face photo of {}, simling",
    "sad": "A face photo of {}, feeling sad",
    "eyeglass": "A face photo of {}, wearing eyeglasses",
}

#"joker": "photo of {}, has a joker nose",
#"red_eye": "photo of {}, red eyes",
#"duck": "photo of {}, making a duck face",
#"open_mouth": "photo of {}, mouth open",

PROMPT_TEMPLATES_CONTROL = {
    #"person": "photo of {}",

    "laughing": "photo of {}, laughing",
    "serious": "photo of {}, serious",
    "smile": "photo of {}, smiling",
    "sad": "photo of {}, looking sad",
    "angry": "photo of {}, angry",
    "surprised": "photo of {}, surprised",
    "beard": "photo of {}, has beard",
    
    "makeup": "photo of {}, with heavy makeup",
    "lipstick": "photo of {}, wearing lipstick",
    "funny": "photo of {}, making a funny face",
    "tongue": "photo of {}, putting the tongue out",

    "singing": "photo of {}, singing with a microphone",
    "cigarette": "photo of {}, smoking, has a cigarette",

    "eyeglass": "photo of {}, wearing eyeglasses",
    "sunglasses": "photo of {}, wearing sunglasses",
}


PROMPT_TEMPLATES_SMALL = {
    "person": "photo of {}"
}


def name2idx(names):
    """Possible names: <num>_*[.jpg][.png] or <num>[.jpg][.png]."""
    return [int(n[:-4].split("_")[0]) for n in names]


class MyStyleDataset(torch.utils.data.Dataset):
    """Load the images of MyStyle dataset"""

    def __init__(self, data_dir="../../data/MyStyle", size=(512, 512),
                 flip_p=0, num_ref=5, seed=None,
                 infer_folder="train", ref_folder="test", mask_folder="test_mask"):
        self.data_dir = data_dir
        self.num_ref = num_ref
        self.size = size
        self.flip_p = flip_p
        self.infer_folder = infer_folder
        self.ref_folder = ref_folder
        self.mask_folder = mask_folder
        self.transform = Compose([Resize(size), ToTensor()])
        self.mask_ds = RandomMaskDataset(size=self.size)
        self.ids = [int(i) for i in os.listdir(data_dir)]
        self.ids.sort()
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, id_idx):
        read_pil = lambda fp: Image.open(open(fp, "rb"))

        id_dir = os.path.join(self.data_dir, f"{id_idx:04d}")

        infer_paths = glob.glob(f"{id_dir}/{self.infer_folder}/*")
        infer_paths.sort()
        #infer_mask_paths = glob.glob(f"{id_dir}/{self.mask_folder}/*.png")
        #infer_mask_paths.sort()
        ref_file_path = os.path.join(id_dir, "ref.txt")
        if os.path.exists(ref_file_path):
            with open(ref_file_path, "r") as f:
                ref_paths = [os.path.join(id_dir, self.ref_folder, f.strip())
                             for f in f.readlines()][:self.num_ref]
        else:
            ref_paths = glob.glob(f"{id_dir}/ref_image/p*.jpg")
            ref_paths.sort()
            ref_paths = ref_paths[:self.num_ref]
        random_masks = torch.stack([self.mask_ds.sample(self.rng)
                                    for _ in infer_paths])
        # load and preprocess the image
        iv_imgs = torch.stack([self.transform(read_pil(fp))
                               for fp in infer_paths])
        rv_imgs = torch.stack([self.transform(read_pil(fp))
                               for fp in ref_paths])
        #mask = torch.stack([self.transform(read_pil(fp))[:1]
        #                    for fp in infer_mask_paths])
        #mask = (mask > 0.5).float()
        num = iv_imgs.shape[0] + rv_imgs.shape[0]
        temps = ["photo of {}" for _ in range(num)]
        return {"infer_image": iv_imgs,
                "ref_image": rv_imgs,
                #"infer_mask": mask.unsqueeze(1),
                "random_mask": random_masks,
                "all_indice": list(range(num)),
                "prompt_template": temps,
                "id": id_idx}


class RandomMaskDataset(torch.utils.data.Dataset):
    """Load a random mask from a folder."""

    def __init__(self, data_dir="../../data/celebahq/mask", size=(256, 256)):
        self.data_dir = data_dir
        self.size = size
        # mask should be stored in PNG format
        self.mask_paths = glob.glob(f"{data_dir}/*.png")
        self.mask_paths.sort()
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.mask_paths)
    
    def sample(self, rng=None):
        if rng is None:
            idx = np.random.randint(0, len(self))
        else:
            idx = rng.randint(0, len(self))
        return self[idx]

    def __getitem__(self, i):
        img = Image.open(self.mask_paths[i])
        if img.size[0] != self.size[0]:
            img = img.resize(self.size, resample=Image.Resampling.NEAREST)
        return self.to_tensor(img)


class CelebAHQIDIDataset(torch.utils.data.Dataset):
    """CelebAHQ IDentity Inpainting dataset.

    Args:
        num_ref: The number of reference images.
        split: train, test, val.
        loop_data: Whether to loop over identity or images in each batch.
            identity: all images of one identity, inference images are masked;
            image-ref: all reference images are masked once, reference selected from ref;
            image-all: all images are masked once, reference selected from both infer and ref.
        single_id: only return data of a single id. For compatibility with Textual Inversion.
    """
    def __init__(self, data_dir="../../data/celebahq", split="train",
                 image_folder="image", ann_folder="annotation",
                 random_mask_dir="../../data/celebahq/mask",
                 num_ref=5, size=(512, 512), flip_p=0, use_caption=True,
                 inpaint_region=["lowerface", "eyebrow", "wholeface"], 
                 loop_data="identity", single_id=None, seed=None):
        self.data_dir = data_dir
        self.split = split
        self.image_folder = image_folder
        self.ann_folder = ann_folder
        self.random_mask_dir = random_mask_dir
        self.num_ref = num_ref
        self.size = size
        self.flip_p = flip_p
        self.use_caption = use_caption
        self.loop_data = loop_data
        self.single_id = single_id
        self.inpaint_regions = inpaint_region
        if type(inpaint_region) is str:
            self.inpaint_regions = [inpaint_region]
        self.num_mask = len(self.inpaint_regions)
        self.rng = np.random.RandomState(seed)
        
        self.transform = Compose([Resize(size), ToTensor()])
        self.bboxes = torch.load(f"{data_dir}/{ann_folder}/region_bbox.pth")["bboxes"]
        n_images = self.bboxes[inpaint_region[0]].shape[0]
        ann_path = f"{data_dir}/{ann_folder}/idi-{num_ref}.json"
        self.ann = json.load(open(ann_path, "r"))

        if os.path.exists(random_mask_dir):
            self.mask_ds = RandomMaskDataset(
                data_dir=random_mask_dir, size=self.size)
        else:
            self.mask_ds = None
        
        self.prompt_templates = PROMPT_TEMPLATES_SMALL #PROMPT_TEMPLATES_LARGE
        self.prepend_text = "A face photo of {}. "
        
        if use_caption:
            caption_path = f"{data_dir}/{ann_folder}/dialog/captions_hq.json"
            caption_dict = json.load(open(caption_path, "r"))
            self.captions = [] # The caption misses on 5380.jpg
            for n in [f"{i}.jpg" for i in range(n_images)]:
                text = self.prepend_text
                if n in caption_dict:
                    text = text + caption_dict[n]["overall_caption"]
                self.captions.append(text)
        else:
            self.captions = [self.prepend_text] * n_images
        self._create_loop_list()

    def _create_loop_list(self):
        split_names = ["train", "test", "val"]
        self.ann["all_ids"] = []
        for k in split_names:
            self.ann["all_ids"] += self.ann[f"{k}_ids"]
        self.ann["all_ids"].sort()
        self.ids = self.ann[f"{self.split}_ids"]
        if self.loop_data == "identity":
            if self.single_id is not None:
                #self.ids = [self.ids[self.single_id]]
                self.ids = [self.single_id]
        elif "image" in self.loop_data:
            key = self.loop_data.split("-")[1] # total, infer, ref
            self.ann["all_images"] = {key: []}
            for k in split_names:
                self.ann["all_images"][key] += self.ann[f"{k}_images"][key]
            self.image_indices = self.ann[f"{self.split}_images"][key]
            if self.single_id is not None:
                self.this_id = self.single_id
                #self.this_id = self.ids[self.single_id]
                m = self.ann["id2image"][self.this_id]
                all_indices = m["infer"] + m["ref"]
                self.image_indices = all_indices if key == "all" else m[key]

    def __len__(self):
        if self.loop_data == "identity":
            return len(self.ids)
        elif "image" in self.loop_data:
            return len(self.image_indices)

    def _read_pil(self, fp):
        fpath = os.path.join(self.data_dir, self.image_folder, fp)
        return Image.open(open(fpath, "rb")).convert("RGB")

    def _fetch_id(self, index):
        if self.loop_data == "identity":
            id_name = self.ids[index]
            id_ann = self.ann["id2image"][self.ids[index]]
            iv_files, rv_files = id_ann["infer"], id_ann["ref"]
        elif "image" in self.loop_data:
            image_name = self.image_indices[index]
            id_name = self.ann["image2id"][name2idx([image_name])[0]]
            id_ann = self.ann["id2image"][id_name]
            iv_files = [image_name]
            if self.loop_data == "image-ref":
                rv_files = [f for f in id_ann["ref"] if f != image_name]
            elif self.loop_data == "image-infer":
                rv_files = id_ann["ref"]
            elif self.loop_data == "image-all":
                all_files = id_ann["infer"] + id_ann["ref"]
                other_files = [f for f in all_files if f != image_name]
                rv_files = list(np.random.choice(
                    other_files, (self.num_ref,), replace=False))
            else:
                raise NotImplementedError
        return iv_files, rv_files, id_name

    def _sample_prompt_temp(self, i):
        """Sample prompt template"""
        rand_temp = np.random.choice(list(self.prompt_templates.values()))
        if not self.use_caption or np.random.rand() < 0.5:
            return rand_temp
        return self.captions[i]

    def __getitem__(self, index):
        iv_files, rv_files, id_idx = self._fetch_id(index)
        # sometimes training pipeline samples according to the order of
        # rv_files, so shuffle here
        self.rng.shuffle(rv_files)
        iv_indices = name2idx(iv_files)
        rv_indices = name2idx(rv_files)
        all_files = iv_files + rv_files
        all_indices = iv_indices + rv_indices
        #print(all_files, all_indices)

        # load and preprocess the image
        iv_imgs = [self._read_pil(fp) for fp in iv_files]
        rv_imgs = [self._read_pil(fp) for fp in rv_files]
        orig_size = iv_imgs[0].size[0]
        scale = float(self.size[0]) / orig_size
        iv_imgs = torch.stack([self.transform(img) for img in iv_imgs])
        rv_imgs = torch.stack([self.transform(img) for img in rv_imgs])
        mask = torch.zeros(len(iv_files), self.num_mask, *iv_imgs.shape[1:])
        for i, gidx in enumerate(iv_indices):
            # obtain and scale the bbox
            for j, rname in enumerate(self.inpaint_regions):
                x_min, y_min, x_max, y_max = (self.bboxes[rname][gidx] * scale).long()
                mask[i, j, :, x_min:x_max, y_min:y_max].fill_(1)
        for i in range(len(iv_imgs)):
            if self.rng.rand() < self.flip_p:
                mask[i] = torch.flip(mask[i], (3,))
                iv_imgs[i] = torch.flip(iv_imgs[i], (2,))
        for i in range(len(rv_imgs)):
            if self.rng.rand() < self.flip_p:
                rv_imgs[i] = torch.flip(rv_imgs[i], (2,))

        indices = self.rng.randint(0, self.num_mask, (iv_imgs.shape[0],))
        selected_mask = torch.stack([mask[i, indices[i]]
                                  for i in range(iv_imgs.shape[0])])
        if self.mask_ds is not None:
            random_masks = torch.stack([self.mask_ds.sample(self.rng)
                for _ in range(iv_imgs.shape[0])])
        else:
            random_masks = selected_mask
        temps = [self._sample_prompt_temp(i) for i in all_indices]
        return {"infer_image": iv_imgs,
                "ref_image": rv_imgs,
                "infer_mask": mask,
                "random_mask": (random_masks + selected_mask).clamp(max=1),
                "all_indice": all_indices,
                "all_file": all_files,
                "prompt_template": temps,
                "id": id_idx}


class IDInpaintDataset(torch.utils.data.Dataset):
    """Load the CelebAHQ inpainting dataset.
    Args:
        data_dir: The data root directory.
        size: The image resolution.
        num_ref: The number of reference views.
        inpaint_region: The name of artificially occluded region.
    """

    def __init__(self, data_dir, size,
                 num_ref=5, inpaint_region=["lowerface", "eyebrow"]):
        self.num_ref = num_ref
        self.inpaint_regions = inpaint_region
        if type(inpaint_region) is str:
            self.inpaint_regions = [inpaint_region]
        self.num_mask = len(self.inpaint_regions)
        self.data_dir = data_dir
        self.size = size
        self.transform = Compose([Resize(size), ToTensor()])
        self.image_paths = glob.glob(f"{data_dir}/image/*")
        self.image_paths.sort()

        self.bboxes = torch.load(f"{data_dir}/annotation/region_bbox.pth")["bboxes"]
        self.region_names = list(self.bboxes.keys())
        self.valid = self._check_valid_bbox()
        self.non_duplicate = torch.ones(len(self.image_paths)).bool()
        with open(f"{data_dir}/annotation/duplicate.txt", "r") as f:
            for l in f.readlines():
                if len(l) < 2:
                    continue
                indices = [int(i) for i in l.split(" ")]
                keep = [i for i in indices if self.valid[i]][0]
                invalid_indices = [i for i in indices if keep != i]
                self.non_duplicate[invalid_indices] = False

        with open(f"{data_dir}/annotation/id2file.json", "r") as f:
            self.id_ann = json.load(f)
        ids = list(self.id_ann.keys())
        self.ids = [i for i in ids if self._check_valid_id(self.id_ann[i])]

    def _check_valid_id(self, image_ids):
        n_non_dup = self.non_duplicate[image_ids].sum()
        n_valid_box = self.valid[image_ids].sum()
        return (n_non_dup >= 1 + self.num_ref) and (n_valid_box >= 1)

    def _check_valid_bbox(self):
        # to change to max(., 0)
        calc_area = lambda a: F.relu(a[:, 2] - a[:, 0]) * F.relu(a[:, 3] - a[:, 1], 0)
        v1 = calc_area(self.bboxes["lowerface"]) > 100
        v2 = calc_area(self.bboxes["eyebrow"]) > 100
        return v1 & v2

    def __len__(self):
        """The total number of samples is the number of identities."""
        return len(self.ids)
    
    def _hflip(self, img, bbox):
        """Horizontal flipping the image and annotation."""
        pass

    def _read_pil(self, fname):
        fpath = f"{self.data_dir}/image/{fname}.jpg"
        return Image.open(open(fpath, "rb"))

    def _fetch_id(self, index):
        identity_index = self.ids[index]
        files = np.array(self.id_ann[identity_index])
        files = files[self.non_duplicate[files]]
        # randomly select an inpainting and a reference view
        valid = self.valid[files].bool()
        valid_indice, invalid_indice = files[valid], files[~valid]
        remain_n = max(0, self.num_ref - invalid_indice.shape[0])
        np.random.RandomState(index).shuffle(valid_indice)
        np.random.RandomState(index).shuffle(invalid_indice)
        rv_gindices = list(valid_indice[:remain_n]) + \
            list(invalid_indice[:self.num_ref])
        iv_gindices = list(valid_indice[remain_n:])
        assert self.valid[iv_gindices].all()
        return iv_gindices, rv_gindices

    def __getitem__(self, index):
        """
        Returns:
            iv_img: (num_inp, 3, H, W)
            rv_imgs: (self.num_ref, 3, H, W)
            mask: (num_inp, self.num_mask, 1, H, W). Region for inpainting is 1, others 0.
        """
        identity_index = self.ids[index]
        files = np.array(self.id_ann[identity_index])
        files = files[self.non_duplicate[files]]
        # randomly select an inpainting and a reference view
        valid = self.valid[files].bool()
        valid_indice, invalid_indice = files[valid], files[~valid]
        remain_n = max(0, self.num_ref - invalid_indice.shape[0])
        np.random.RandomState(index).shuffle(valid_indice)
        np.random.RandomState(index).shuffle(invalid_indice)
        rv_gindices = list(valid_indice[:remain_n]) + \
            list(invalid_indice[:self.num_ref])
        iv_gindices = list(valid_indice[remain_n:])
        assert self.valid[iv_gindices].all()
        all_indices = iv_gindices + rv_gindices
        # load and preprocess the image
        iv_imgs = [self._read_pil(i) for i in iv_gindices]
        rv_imgs = [self._read_pil(i) for i in rv_gindices]
        orig_size = iv_imgs[0].size[0]
        scale = float(self.size[0]) / orig_size
        iv_imgs = torch.stack([self.transform(img) for img in iv_imgs])
        rv_imgs = torch.stack([self.transform(img) for img in rv_imgs])
        mask = torch.zeros(len(iv_gindices), self.num_mask, *iv_imgs.shape[1:])
        for i, gidx in enumerate(iv_gindices):
            # obtain and scale the bbox
            for j, rname in enumerate(self.inpaint_regions):
                x_min, y_min, x_max, y_max = (self.bboxes[rname][gidx] * scale).long()
                mask[i, j, :, x_min:x_max, y_min:y_max].fill_(1)
        return iv_imgs, rv_imgs, mask, all_indices


class InpaintDataset(torch.utils.data.Dataset):
    """Load the inpainting dataset (for no reference).
    Args:
        data_dir: The data root directory.
        size: The image resolution.
        num_ref: The number of reference views.
        inpaint_region: The name of artificially occluded region.
    """

    def __init__(self, data_dir,
            image_folder="image",
            ann_folder="annotation",
            size=[256, 256],
            inpaint_region="random"):
        self.data_dir = data_dir
        self.image_folder = image_folder
        self.ann_folder = ann_folder
        self.inpaint_region = inpaint_region
        self.data_dir = data_dir
        self.size = size
        self.transform = Compose([Resize(size), ToTensor()])
        self.image_paths = glob.glob(f"{data_dir}/{image_folder}/*")
        self.image_paths.sort()
        self.bboxes = torch.load(f"{data_dir}/{ann_folder}/bbox.pth")["bboxes"]
        self.region_names = list(self.bboxes.keys())
        self.valid = self._check_valid_bbox()

    def _check_valid_bbox(self):
        # to change to max(., 0)
        calc_area = lambda a: F.relu(a[:, 2] - a[:, 0]) * F.relu(a[:, 3] - a[:, 1], 0)
        v1 = calc_area(self.bboxes["lowerface"]) > 100
        v2 = calc_area(self.bboxes["eyebrow"]) > 100
        return v1 & v2

    def __len__(self):
        """The total number of samples is the number of identities."""
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """
        Returns:
            iv_img: (3, H, W)
            mask: (3, H, W)
        """
        # load and preprocess the image
        iv_img = Image.open(open(self.image_paths[index], "rb"))
        orig_size = iv_img.size[0]
        scale = float(self.size[0]) / orig_size
        iv_img = self.transform(iv_img)

        # randomly select an inpainting region
        if self.inpaint_region == "random":
            rname = np.random.choice(self.region_names, 1)[0]
        else:
            rname = self.inpaint_region
        # obtain and scale the bbox
        n = self.image_paths[index]
        bbox_idx = int(n[n.rfind("/")+1:n.rfind(".")])
        x_min, y_min, x_max, y_max = (self.bboxes[rname][bbox_idx] * scale).long()
        # convert bbox to mask
        mask = torch.ones_like(iv_img)
        mask[:, x_min:x_max, y_min:y_max] = 0
        return iv_img, mask


class SimpleDataset(torch.utils.data.Dataset):
    """
    Image-only datasets.
    """
    def __init__(self, data_path, size=None, transform=ToTensor()):
        self.size = size
        self.data_path = data_path
        self.transform = transform

        self.files = sum([[file for file in files if ".jpg" in file or ".png" in file] for path, dirs, files in os.walk(data_path) if files], [])
        self.files.sort()

    def __getitem__(self, idx):
        fpath = self.files[idx]
        with open(os.path.join(self.data_path, fpath), "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.size:
                img = img.resize(self.size, Image.BILINEAR)
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.files)


class SimpleMaskDataset(torch.utils.data.Dataset):
    """
    The mask and image are stored under the same folder with naming format.
    00003_crop000_mask000
    00003
    """
    def __init__(self, data_path, size=None, transform=ToTensor()):
        self.size = size
        self.data_path = data_path
        self.transform = transform

        files = sum([[file for file in files if ".jpg" in file or ".png" in file] for path, dirs, files in os.walk(data_path) if files], [])
        files.sort()
        self.image_files = files[::2]
        self.mask_files = files[1::2]

    def _read_resize_pil(self, fpath, resample="bilinear"):
        with open(os.path.join(self.data_path, fpath), "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.size and img.size[0] != self.size[0]:
                flag = Image.BILINEAR if resample == "bilinear" else Image.NEAREST
                img = img.resize(self.size, flag)
        return img

    def __getitem__(self, idx):
        image = self._read_resize_pil(self.image_files[idx], "bilinear")
        mask = self._read_resize_pil(self.mask_files[idx], "nearest")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask[:1] # only need the first channel

    def __len__(self):
        return len(self.image_files)