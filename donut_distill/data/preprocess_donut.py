import json
from os import listdir, path
from pathlib import Path
import shutil
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Union

def preprocess_annotations_links_funsd(annotation_path: str) -> Dict[str, List[Dict[str, Union[str, List[str]]]]]:
    """
    Preprocesses FUNSD annotation files by extracting question-answer pairs.

    This function:
    - Reads and parses the JSON annotation file.
    - Maps field IDs to their corresponding text and labels.
    - Extracts questions and finds their linked answers.
    - Returns a structured dictionary of question-answer pairs.

    Args:
        annotation_path (str): Path to the JSON annotation file.

    Returns:
        Dict[str, List[Dict[str, Union[str, List[str]]]]]: A dictionary containing extracted form data in the format:
        {
            "form": [
                {"question": "What is your name?", "answer": "John Doe"},
                {"question": "What are the ingredients?", "answers": ["Sugar", "Flour", "Butter"]},
                ...
            ]
        }
    """
    # Load annotation data from JSON file
    with open(annotation_path, "r") as f:
        data = json.load(f)

    # Extract form fields from the annotation data
    fields = data.get("form", [])

    # Create a mapping of field IDs to their corresponding text and labels
    id_to_text = {
        field["id"]: {
            "text": field["text"],
            "label": field["label"],
        }
        for field in fields
    }

    qa_pairs = []
    for field in fields:
        question_text = field["text"]
        links = field.get("linking", [])

        # Only process fields labeled as "question"
        if field["label"] != "question":
            continue

        answers = []
        for link in links:
            if field["id"] in link:
                link.remove(field["id"])  # Remove the current question's ID from the link

            # Ensure the linked field is labeled as "answer"
            if id_to_text[link[0]]["label"] != "answer":
                continue

            answers.append(id_to_text[link[0]]["text"])

        # Store question-answer pairs in the appropriate format
        if len(answers) == 0:
            qa_pairs.append({"question": question_text, "answer": ""})  # No answer found
        elif len(answers) == 1:
            qa_pairs.append({"question": question_text, "answer": answers[0]})  # Single answer
        else:
            qa_pairs.append({"question": question_text, "answers": answers})  # Multiple answers

    return {"form": qa_pairs}


def preprocess_annotations_labels_funsd(annotation_path: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Preprocesses FUNSD annotation files by extracting text-label pairs.

    This function:
    - Reads and parses the JSON annotation file.
    - Extracts the text and label fields from each form element.
    - Returns a structured dictionary of extracted elements.

    Args:
        annotation_path (str): Path to the JSON annotation file.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing extracted elements in the format:
        {
            "elements": [
                {"text": "John Doe", "label": "name"},
                {"text": "123 Main St", "label": "address"},
                ...
            ]
        }
    """
    # Load annotation data from JSON file
    with open(annotation_path, "r") as f:
        data = json.load(f)

    # Extract form fields from the annotation data
    fields = data.get("form", [])

    result = []
    for field in fields:
        text = field["text"]
        label = field["label"]
        result.append({"text": text, "label": label})  # Store extracted text and label

    return {"elements": result}



def preprocess_directory_funsd(
    directory_path: str, 
    output_path: str, 
    process_annotation_fn: Callable[[str], dict] = preprocess_annotations_links_funsd, 
    max_datapoints: Optional[int] = None
):
    """
    Preprocesses a directory containing FUNSD dataset images and annotations.

    This function:
    - Creates the output directory if it doesn’t exist.
    - Reads images and their corresponding annotation files.
    - Processes annotations using the provided function.
    - Saves metadata in a JSONL file.
    - Copies images to the output directory.

    Args:
        directory_path (str): Path to the input directory containing "images" and "annotations" subdirectories.
        output_path (str): Path to store processed images and metadata.
        process_annotation_fn (Callable[[str], dict], optional): Function to process annotations. Defaults to preprocess_annotations_links_funsd.
        max_datapoints (Optional[int], optional): Maximum number of datapoints to process. If None, all files are processed.

    Returns:
        None
    """
    # Create output directory if it does not exist
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Open metadata file for writing
    metadata_file = open(path.join(output_path, "metadata.jsonl"), "w")

    # Define paths for images and annotations
    image_directory = path.join(directory_path, "images")
    annotation_directory = path.join(directory_path, "annotations")

    # Get list of file IDs by removing the '.png' extension from image filenames
    file_ids = [
        file.removesuffix(".png")
        for file in listdir(image_directory)
        if path.isfile(path.join(image_directory, file))
    ]

    # Iterate through files, processing annotations and copying images
    for i, file_id in enumerate(file_ids):
        if max_datapoints and i >= max_datapoints:
            break  # Stop processing if the maximum number of datapoints is reached

        image_path = path.join(image_directory, f"{file_id}.png")
        annotation_path = path.join(annotation_directory, f"{file_id}.json")

        # Process the annotation file
        preprocessed_annotation = process_annotation_fn(annotation_path)

        # Store the preprocessed annotation in metadata format
        gt = {"gt_parse": preprocessed_annotation}
        file_metadata = {
            "file_name": f"{file_id}.png",
            "ground_truth": json.dumps(gt),
        }
        metadata_file.write(json.dumps(file_metadata) + "\n")  # Write metadata to file

        # Copy the image to the output directory
        shutil.copy(image_path, output_path)

    # Close the metadata file
    metadata_file.close()


def preprocess_docvqa(
    annotations_path: str, 
    images_path: str, 
    output_path: str, 
    train_limit: Optional[int] = None, 
    validation_limit: Optional[int] = None
):
    """
    Preprocesses the DocVQA dataset by extracting questions and answers, 
    and organizes them into training and validation splits.

    This function:
    - Reads annotation files and extracts questions and answers.
    - Applies dataset limits if specified.
    - Organizes images and metadata into structured output directories.
    - Copies relevant images to the output location.

    Args:
        annotations_path (str): Path to the directory containing JSON annotation files.
        images_path (str): Path to the directory containing images.
        output_path (str): Path to the directory where processed data will be saved.
        train_limit (Optional[int], optional): Maximum number of training samples to process. Defaults to None.
        validation_limit (Optional[int], optional): Maximum number of validation samples to process. Defaults to None.

    Returns:
        None
    """
    for annotation_file_name in listdir(annotations_path):
        annotation_file_path = path.join(annotations_path, annotation_file_name)
        
        if path.isdir(annotation_file_path) or not annotation_file_path.endswith(".json"):
            continue

        # Load annotation file
        with open(annotation_file_path, "r") as annotation_file:
            annotation_data = json.load(annotation_file)

        dataset_split = annotation_data["dataset_split"]

        # Skip test set, since it doesn't have answers
        if dataset_split == "test":
            continue

        data = annotation_data["data"]

        # Apply dataset limits if specified
        if dataset_split == "train" and train_limit is not None:
            data = data[:train_limit]
        elif dataset_split == "val" and validation_limit is not None:
            data = data[:validation_limit]

        # Create output directory for the dataset split
        output_directory = path.join(output_path, dataset_split)
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        # Open metadata file for writing
        metadata_file = open(path.join(output_directory, "metadata.jsonl"), "w")

        # Process each data point
        for datapoint in tqdm(data, desc=f"Processing {dataset_split} set"):
            image_path = path.join(images_path, datapoint["image"])
            image_name = path.basename(image_path)

            question = datapoint["question"]
            answers = datapoint["answers"]

            # Create ground truth parses (each answer is stored separately)
            gt_parses = [{"question": question, "answer": answer} for answer in answers]

            # Store metadata for this sample
            file_metadata = {
                "file_name": image_name,
                "ground_truth": json.dumps({"gt_parses": gt_parses}),
            }

            metadata_file.write(json.dumps(file_metadata) + "\n")  # Write to metadata file

            # Copy the corresponding image to the output directory
            shutil.copy(image_path, output_directory)

        # Close the metadata file after processing all samples
        metadata_file.close()


if __name__ == "__main__":
    # test_directory = "dataset/testing_data"
    # train_directory = "dataset/training_data"
    # process_annotation_fn=preprocess_annotations_labels_funsd
    # preprocess_directory(test_directory, "preprocessed_dataset/test", process_annotation_fn)
    # preprocess_directory(train_directory, "preprocessed_dataset/train", process_annotation_fn)

    # preprocess_directory(train_directory, "preprocessed_dataset/test", process_annotation_fn, max_datapoints=5)
    # preprocess_directory(train_directory, "preprocessed_dataset/train", process_annotation_fn, max_datapoints=5)

    # preprocess_directory_funsd(train_directory, "preprocessed_dataset/test", process_annotation_fn)
    # preprocess_directory_funsd(train_directory, "preprocessed_dataset/train", process_annotation_fn)

    # preprocess_docvqa("docvqa/queries", "docvqa", "preprocessed_dataset_docvqa_small", 50, 10)
    preprocess_docvqa("docvqa/queries", "docvqa", "preprocessed_dataset_docvqa")


