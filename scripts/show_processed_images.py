import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage
from donut_distill.donut_dataset import DonutDataset
import donut_distill.config as CONFIG
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)

def show_preprocessed_image(pixel_values, image_mean, image_std, rescale_factor=None):
    """
    Visualize an image after reversing preprocessing.

    Args:
        pixel_values (torch.Tensor): The preprocessed image tensor (C, H, W).
        image_mean (List[float]): Mean used for normalization.
        image_std (List[float]): Standard deviation used for normalization.
        rescale_factor (float, optional): Factor used to rescale the image. Default is None.
    """
    # Ensure tensor is in (C, H, W) format
    if len(pixel_values.shape) != 3 or pixel_values.shape[0] not in [1, 3]:
        raise ValueError("Unexpected tensor shape for pixel values. Expected (C, H, W).")

    # Reverse normalization
    mean = torch.tensor(image_mean).view(-1, 1, 1)
    std = torch.tensor(image_std).view(-1, 1, 1)
    denormalized = pixel_values * std + mean

    # Reverse rescale if applied
    # if rescale_factor:
    #     denormalized *= rescale_factor

    # Clip values to valid range [0, 255]
    denormalized = torch.clamp(denormalized, 0, 255)

    # Convert to PIL image
    pil_image = ToPILImage()(denormalized)

    # Display the image
    plt.imshow(pil_image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    donut_config = VisionEncoderDecoderConfig.from_pretrained(CONFIG.MODEL_ID)
    donut_config.encoder.image_size = CONFIG.INPUT_SIZE
    donut_config.decoder.max_length = CONFIG.MAX_LENGTH

    processor = DonutProcessor.from_pretrained(CONFIG.MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(
        CONFIG.MODEL_ID, config=donut_config
    )

    processor.image_processor.size = CONFIG.INPUT_SIZE[::-1]
    processor.image_processor.do_align_long_axis = False
    train_dataset = DonutDataset(
        dataset_name_or_path=CONFIG.DATASET,
        processor=processor,
        model=model,
        max_length=CONFIG.MAX_LENGTH,
        split=CONFIG.DATASET_NAME_VALIDATE,
        task_start_token="<s_funsd>",
        sort_json_key=False,  # cord dataset is preprocessed, so no need for this
    )

    # show_image_from_dataset(train_dataset, 0, processor)
    # help(processor.image_processor)
    # for i in range(5):
    #     train_dataset[i]
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std
    rescale_factor = processor.image_processor.rescale_factor  # Might be None if not used
    size = processor.image_processor.size  # For resizing
    print(image_mean, image_std, rescale_factor, size)

    for i in range(5):
        # Get preprocessed image from dataset
        pixel_values, _, _,  _ = train_dataset[0]  # Adjust indexing as needed

        # Visualize
        show_preprocessed_image(pixel_values, image_mean, image_std, rescale_factor)
        pil_image = ToPILImage()(pixel_values)

        # Display the image
        plt.imshow(pil_image)
        plt.axis("off")
        plt.show()
