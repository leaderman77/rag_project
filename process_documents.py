import os
import pickle
from load_papers import PyMuPDFReader


def process_documents(input_dir, save_path):
    """
    Processes documents by either loading them from a save path if they exist,
    or by reading and cleaning them from an input directory, then saving to the save path.

    Parameters:
    input_dir (str): Directory where the documents are stored.
    save_path (str): File path where processed documents are to be saved.

    Returns:
    list: A list of processed documents.
    """
    # Check if the processed documents already exist in the save path
    if not os.path.exists(save_path):
        # Load documents from the input directory
        documents = PyMuPDFReader(input_dir=input_dir).load_data(show_progress=True, raise_on_error=False)
        # Clean the loaded documents
        documents = clean_data(documents)
        # Save the cleaned documents
        save_documents(documents, save_path)
    else:
        # Load documents from the save file
        documents = load_documents_from_file(save_path)

    return documents


def save_documents(documents, save_path):
    """
    Saves documents to a file using pickle.

    Parameters:
    documents (list): Documents to save.
    save_path (str): Path where the documents should be saved.
    """
    with open(save_path, 'wb') as file:
        pickle.dump(documents, file)
    print(f"Documents saved to {save_path}")


def load_documents_from_file(file_path):
    """
    Loads documents from a pickle file.

    Parameters:
    save_path (str): Path to the file from which documents are to be loaded.

    Returns:
    list: Loaded documents.
    """
    with open(file_path, 'rb') as file:
        documents = pickle.load(file)
    return documents


def clean_data(documents):
    """
    Placeholder function to clean data.

    Parameters:
    documents (list): List of documents to be cleaned.

    Returns:
    list: Cleaned documents.
    """
    cleaned_documents = remove_references_from_documents(documents)
    return cleaned_documents


def remove_references_from_documents(documents):
    """
    Remove the references section from the text of each document, preserving sections like "Appendix".

    Parameters:
    documents (list): List of concatenated document dictionaries.

    Returns:
    cleaned_documents (list): List of documents with references removed.
    """

    # Keywords to identify the start of the references section
    reference_markers = ["\nReferences\n", "\nREFERENCES", "References\n", "Bibliography\n"]

    # Keywords that indicate sections that should be preserved
    preservation_markers = ["Appendix",
                            "Complete Results for Top- pand Top- kDecoding",
                            "\nA A DDITIONAL ANALYSIS\n",
                            "Supplemental Material\n",
                            "Question Answering\n",
                            "Additional Material\n",
                            "Additional Results: Relevant Document Scaling\n",
                            "BERT Finetuning Hyperparameters\n",
                            "ADDITIONAL ANALYSIS\n",
                            "Development Set Results\n"]
                            #"Analytic FIM for two-layer model\n"]

    for doc in documents:
        text = doc.text

        # Find the position of the references section
        ref_start_idx = None
        for marker in reference_markers:
            ref_start_idx = text.find(marker)
            if ref_start_idx != -1:
                break

        if ref_start_idx != -1:
            # Check if any preservation markers are found after references
            remaining_text = text[ref_start_idx:]
            preserve_start_idx = None
            for preserve_marker in preservation_markers:
                preserve_start_idx = remaining_text.find(preserve_marker)
                if preserve_start_idx != -1:
                    preserve_start_idx += ref_start_idx  # Adjust index relative to the entire text
                    break

            if preserve_start_idx != -1:
                # Keep text up to the start of the preserve marker
                cleaned_text = text[:ref_start_idx] + "\n" + text[preserve_start_idx:]
            else:
                # No preserve marker found; remove everything after references
                cleaned_text = text[:ref_start_idx].strip()
        else:
            # If no references found, keep the text as it is
            cleaned_text = text

        doc.text = cleaned_text

    return documents