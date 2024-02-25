import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import networkx as nx
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import (Compose, ConvertImageDtype,
                                    InterpolationMode, Lambda, Resize)
from tqdm import tqdm
from typing_extensions import Literal


def read_json(file_path: str):
    with open(str(file_path)) as json_data:
        return json.loads(json_data.read())


class ImageFilter():
    def __init__(
        self,
        path_to_data: Path,
        segmentation: bool = False,
        perceptual_similarity: bool = False,
        perceptual_threshold: float = 0.4,
        segmentation_path: Literal['segmentation', 'mask'] = 'segmenation',
        num_objects: Optional[int] = None
    ):

        self.perceptual_similarity = perceptual_similarity
        self.num_objects = num_objects
        self.segmentation = segmentation
        self.segmentation_path = segmentation_path

        self.object_dict = {}
        self._blacklist = []

        if self.perceptual_similarity:
            self.max_indepent_set(path_to_data, threshold=perceptual_threshold)

        if self.num_objects:
            self.object_limit(path_to_data)

    def max_indepent_set(self, path_to_data: Path, threshold: float):
        """Function reads all similarity files and computes the largest
        independent set of nodes without direct connection (close
        similarity metric) between each other.

        Args:
            path_to_data (Path): Path to "data" folder
            threshold (float): Threshold for similar images
        """
        # path to all .json files
        similarity_path = [x for x in Path(
            path_to_data).glob('**/similarity/*.json')]

        similarities = []

        # loop over all similarities
        for file_path in similarity_path:

            folder1_name = file_path.parent.parent.name
            img1_name = file_path.name.split(".json")[0]

            d = read_json(file_path)

            for folder2_name, data in d.items():
                for img2_name, scores in data.items():

                    similarity_tuple = (
                        folder1_name+"/"+img1_name, folder2_name+"/"+img2_name)

                    # requirement
                    if scores["Perceptual Similarity"] > threshold:
                        similarities.append(similarity_tuple)

        # create graph
        G = nx.Graph(similarities)

        # compute set
        max_independent_nodes = nx.maximal_independent_set(G)
        remaining = [node for node in list(
            G.nodes) if node not in max_independent_nodes]

        # add remaining nodes (not in max independet set) to blacklist
        self._blacklist.extend(remaining)

    def object_limit(self, path_to_data: Path):
        """Creates dictionary for number of items in images

        Args:
            path_to_data (Path): Path to "data" folder
        """

        object_path = [x for x in Path(
            path_to_data).glob('**/object_types.json')]

        for obj in object_path:
            d = read_json(str(obj))

            folder_name = obj.parent.name
            for img, object_metadata in d.items():

                self.object_dict[folder_name+"/" +
                                 img] = object_metadata['TotalObjects']

    def __call__(self, image_path: Path):

        folder_name = image_path.parent.parent.name
        image_name = image_path.name.split("_")[0]
        name = folder_name+"/"+image_name

        # check if number of objects in scene exceed the threshold
        if self.num_objects:
            if self.object_dict[name] > self.num_objects:
                return False

        # check if perceptual_similarity exceeds threshold
        if self.perceptual_similarity:
            if name in self._blacklist:
                return False

        # check if segmenation file exists
        if self.segmentation:
            segmentation_file = image_path.parent.parent / \
                self.segmentation_path / image_path.name
            if not segmentation_file.is_file():
                return False

        return True


class Procthor(Dataset):

    """The Prothor dataset is a dataset generated using https://github.com/allenai/ai2thor.
    """

    def __init__(
        self,
        path_to_data,
        transform_image: Optional[Callable[[Tensor], Tensor]] = None,
        transform_segmentation: Optional[Callable[[Tensor], Tensor]] = None, 
        image_pair: bool = False,
        segmentation: bool = False, 
        segmentation_path: Literal['segmentation', 'mask'] = 'segmenation',
        max_depth: Optional[float] = None,
        file_ending: str = 'first',
        debug: bool = False,
        size: int = None,
        **kwargs
    ):

        self.path_to_data = path_to_data
        self.transform_image = transform_image
        self.segmentation = segmentation
        self.segmentation_path = segmentation_path
        self.transform_segmentation = transform_segmentation
        self.max_depth = max_depth
        self.image_pair = image_pair
        self.filter_obj = ImageFilter(
            path_to_data,
            segmentation=self.segmentation,
            segmentation_path=self.segmentation_path,
            **kwargs
            )

        # file_ending is either empty string or 'first' to indicate wether
        # pairs of images are used
        print("Initialize dataset")
        pbar = tqdm(Path(self.path_to_data).glob(
            f'{"proc_*000" if debug else "**"}/color/*{file_ending}.png'))
        self.files = [x for x in pbar if self.filter_obj(x)]

        if size:
            self.files = self.files[:size]

        if self.image_pair:
            self.add_pair()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        image_path = self.files[idx]

        if self.image_pair:
            image = (read_image(str(image_path[0])), read_image(
                str(image_path[1])))

            if self.transform_image:
                image = (self.transform_image(
                    image[0]), self.transform_image(image[1]))

            image_path = image_path[0]  # for getting segmenation

        else:
            image = read_image(str(image_path))

            if self.transform_image:
                image = self.transform_image(image)

        if self.segmentation:
            instance_segmentation = read_image(
                str(image_path.parent.parent / self.segmentation_path /
                    image_path.name))

            if self.max_depth:
                depth = read_image(
                    str(image_path.parent.parent / "depth" / image_path.name))

                background = depth > self.max_depth

                if torch.sum(background) > 0:
                    background_value = instance_segmentation[background][0]
                    instance_segmentation[background] = background_value

            if self.transform_segmentation:
                instance_segmentation = self.transform_segmentation(
                    instance_segmentation)

            return instance_segmentation, image

        return image

    def add_pair(self):
        print("Find pairs")
        pbar = tqdm(enumerate(self.files))
        for i, path in pbar:
            name = path.name.split('_')[0]
            gen = path.parent.glob(f'{name}*.png')
            self.files[i] = [x for x in gen]


class CLEVR(Dataset):

    def __init__(
        self,
        path_to_data,
        transform_image: Optional[Callable[[Tensor], Tensor]] = None,
        limit: Optional[int] = None
    ):

        self.path_to_data = path_to_data
        self.transform_image = transform_image

        self.files = [x for x in Path(self.path_to_data).glob(
            'output_3_to_6_mirror/images/*.png')]

        if limit:
            self.files = self.files[:limit]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        image_path = self.files[idx]
        # alpha channels seems to be constant
        image = read_image(str(image_path))[:3]

        if self.transform_image:
            image = self.transform_image(image)

        return image


def create_dataset(
    task_name: str,
    path_to_data: str,
    image_size: Tuple[float, float],
    transform_image: Optional[Callable[[Tensor], Tensor]] = None,
    transform_segmentation: Optional[Callable[[Tensor], Tensor]] = None, 
    segmentation: bool = True,
    file_ending: str = "first",
    image_pair: bool = False,
    debug: bool = False,
    size: int = None,
    segmentation_path='segmentation',
    **filter_kwargs
):

    if transform_image is None:
        transform_image = Compose([
            Resize(image_size),
            ConvertImageDtype(dtype=torch.float32)
        ])

    if transform_segmentation is None:
        transform_segmentation = Resize(
            image_size, interpolation=InterpolationMode.NEAREST)

    if task_name == "procthor":
        return Procthor(
            path_to_data=path_to_data,
            transform_image=transform_image,
            transform_segmentation=transform_segmentation,
            segmentation=segmentation,
            file_ending=file_ending,
            image_pair=image_pair,
            debug=debug, size=size,
            segmentation_path=segmentation_path,
            **filter_kwargs
            )

    elif task_name == 'clevr':
        return CLEVR(
            path_to_data=path_to_data,
            transform_image=transform_image
            )

    else:
        raise ValueError("Invalid name for task type.")
