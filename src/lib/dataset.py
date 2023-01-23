"""Datasets."""
import torch, glob, json, os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from PIL import Image


PROMPT_TEMPLATES_LARGE = {
    "1": "a photo of {}",
    "2": "a photo of the {}",
    "3": "a good photo of {}",
    "4": "a cropped photo of {}",
    "5": "a close-up photo of {}",
    "6": "a bright photo of {}",
    "7": "a face photo of {}",
    "8": "a face photo of the {}",
    "9": "a face photo of cool {}",
    "10": "a beautiful face photo of {}",
}


PROMPT_TEMPLATES_MEDIUM = {
    "person": "A face photo of {}",
    "smile": "A face photo of {}, simling",
    "sad": "A face photo of {}, feeling sad",
    "eyeglass": "A face photo of {}, wearing eyeglasses"
}

PROMPT_TEMPLATES_SMALL = {
    "person": "A face photo of {}"
}



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
    
    def sample(self):
        return self[np.random.randint(0, len(self))]

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
    def __init__(self, data_dir="../../data/celebahq",
                 image_folder="image", ann_folder="annotation",
                 size=(256, 256), num_ref=5,
                 inpaint_region=["lowerface", "eyebrow"],
                 split="train", loop_data="identity", single_id=None):
        self.loop_data = loop_data
        self.single_id = single_id
        self.data_dir = data_dir
        self.image_folder = image_folder
        self.ann_folder = ann_folder
        self.size = size
        self.num_ref = num_ref
        self.inpaint_regions = inpaint_region
        if type(inpaint_region) is str:
            self.inpaint_regions = [inpaint_region]
        self.num_mask = len(self.inpaint_regions)
        self.split = split
        self.transform = Compose([Resize(size), ToTensor()])
        self.bboxes = torch.load(f"{data_dir}/{ann_folder}/region_bbox.pth")["bboxes"]
        ann_path = f"{data_dir}/{ann_folder}/celebahq-idi-{num_ref}.json"
        self.ann = json.load(open(ann_path, "r"))
        self._create_loop_list()

    def _create_loop_list(self):
        split_names = ["train", "test", "val"]
        self.ann["all_ids"] = []
        for k in split_names:
            self.ann["all_ids"] += self.ann[f"{k}_ids"]
        self.ann["all_ids"].sort()
        
        if self.loop_data == "identity":
            self.ids = self.ann[f"{self.split}_ids"]
            if self.single_id is not None:
                self.ids = [self.single_id]
        elif "image" in self.loop_data:
            key = self.loop_data.split("-")[1] # total, infer, ref
            self.ann["all_images"] = {key: []}
            for k in split_names:
                self.ann["all_images"][key] += self.ann[f"{k}_images"][key]
            self.image_indices = self.ann[f"{self.split}_images"][key]
            if self.single_id is not None:
                self.this_id = int(self.single_id)
                self.image_indices = self.ann["id2image"][str(self.this_id)][key]

    def __len__(self):
        if self.loop_data == "identity":
            return len(self.ids)
        elif "image" in self.loop_data:
            return len(self.image_indices)

    def _read_pil(self, fname):
        fpath = f"{self.data_dir}/image/{fname}.jpg"
        return Image.open(open(fpath, "rb"))

    def _fetch_id(self, index):
        if self.loop_data == "identity":
            id_idx = self.ids[index]
            id_ann = self.ann["id2image"][str(self.ids[index])]
            iv_indices, rv_indices = id_ann["infer"], id_ann["ref"]
        elif "image" in self.loop_data:
            image_idx = self.image_indices[index]
            id_idx = self.ann["image2id"][image_idx]
            id_ann = self.ann["id2image"][str(id_idx)]
            iv_indices = [image_idx]
            if self.loop_data == "image-ref":
                rv_indices = [i for i in id_ann["ref"] if i != image_idx]
            elif self.loop_data == "image-infer":
                rv_indices = id_ann["ref"]
            elif self.loop_data == "image-all":
                all_indices = id_ann["infer"] + id_ann["ref"]
                other_indices = [i for i in all_indices if i != image_idx]
                rv_indices = list(np.random.choice(
                    other_indices, (self.num_ref,), replace=False))
            else:
                raise NotImplementedError
        return iv_indices, rv_indices, id_idx

    def __getitem__(self, index):
        iv_indices, rv_indices, id_idx = self._fetch_id(index)
        # load and preprocess the image
        iv_imgs = [self._read_pil(i) for i in iv_indices]
        rv_imgs = [self._read_pil(i) for i in rv_indices]
        orig_size = iv_imgs[0].size[0]
        scale = float(self.size[0]) / orig_size
        iv_imgs = torch.stack([self.transform(img) for img in iv_imgs])
        rv_imgs = torch.stack([self.transform(img) for img in rv_imgs])
        mask = torch.zeros(len(iv_indices), self.num_mask, *iv_imgs.shape[1:])
        for i, gidx in enumerate(iv_indices):
            # obtain and scale the bbox
            for j, rname in enumerate(self.inpaint_regions):
                x_min, y_min, x_max, y_max = (self.bboxes[rname][gidx] * scale).long()
                mask[i, j, :, x_min:x_max, y_min:y_max].fill_(1)
        return {"infer_image": iv_imgs,
                "ref_image": rv_imgs,
                "infer_mask": mask,
                "all_indice": iv_indices + rv_indices,
                "id": id_idx}


class RTMCelebAHQIDIDataset(CelebAHQIDIDataset):
    """Add random masks and language to images."""
    def __init__(self, flip_p=0.5, special_token="*",
            prmopt_templates=PROMPT_TEMPLATES_LARGE,
            load_id_feat=False,
            *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flip_p = flip_p
        self.special_token = special_token
        self.templates = prmopt_templates
        self.to_tensor = ToTensor()
        self.mask_ds = RandomMaskDataset(size=self.size)
        self.load_id_feat = load_id_feat
        if load_id_feat:
            self.id_feat = torch.load(f"{self.data_dir}/annotation/id_feat.pth", map_location="cpu")

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        """
        Args:
            iv_imgs: inference images.
            iv_masks: The semantic inpainting mask. Used in official testing.
            random_masks: Randomly generated masks. Used in training.
            id_feats: The face identity features of reference images.
                      Used for saving computation.
            rv_imgs: The reference images.
            p_names: The short-hand name of prompts.
            prompts: Randomly generated prompts.
        """
        data = super().__getitem__(index)
        n_infer = data["infer_image"].shape[0]
        random_masks = torch.stack([
            self.mask_ds.sample() for _ in range(n_infer)])
        p_names = [np.random.choice(list(self.templates.keys()))
            for _ in range(n_infer)]
        prompts = [self.templates[key].format(self.special_token)
            for key in p_names]
        data.update({
            "random_mask": random_masks, # (n_infer, 3, H, W)
            "p_name": p_names, "prompt": prompts})
        # The feature from reference images
        if self.load_id_feat:
            data["ref_id_feat"] = self.id_feat[data["all_indice"][n_infer:]]
            data["infer_id_feat"] = self.id_feat[data["all_indice"][:n_infer]]
        return data


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

        with open(f"{data_dir}/annotation/celebahq_id2file.json", "r") as f:
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